# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from . import MOTMODEL_REGISTRY

from .base import MOTDINOBASE
import torch
import copy
from collections import deque
from torch import nn
from motlib.mot_models.structures import Instances
from motlib.mot_models.network.dino_mot.tracker.util import RuntimeTrackerBase, TrackerPostProcess


__all__ = ['MOTDINO']


@MOTMODEL_REGISTRY.register()
class MOTDINO(MOTDINOBASE):
    def __init__(self, args, backbone, transformer, criterion, query_interaction_layer):
        super().__init__()
        self.args = args
        self.dino = self.init_dino(backbone, transformer)
        self.criterion = criterion
        self.mem_bank_len = self.args.mem_bank_len if hasattr(self.args, "mem_bank_len") else 0
        self.track_embed = query_interaction_layer

        self.num_classes = self.criterion.num_classes

        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase(self.args)
    
    def clear(self):
        self.track_base.clear()
    
    def _generate_empty_tracks(self, num_queries=None):
        track_instances = Instances((1, 1))
        query_embed = self.dino.transformer.tgt_embed
        num_queries_raw, dim = query_embed.weight.shape
        device = query_embed.weight.device

        if self.args.amp:
            dtype = torch.float16
        else:
            dtype = torch.float

        if num_queries is None:
            num_queries = num_queries_raw

        track_instances.ref_pts = torch.zeros((num_queries, 4), device=device)
        track_instances.query_pos = torch.zeros((num_queries, dim), device=device)
        track_instances.output_embedding = torch.zeros((num_queries, dim), device=device)

        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=dtype, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=dtype,device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=dtype, device=device)
        track_instances.strack = [None for _ in range(len(track_instances))]
        track_instances.track_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = [deque(maxlen=mem_bank_len) for _ in range(len(track_instances))]
        track_instances.ref_bank = [deque(maxlen=mem_bank_len) for _ in range(len(track_instances))]

        return track_instances.to(device)
    
    def _forward_single_image(self, memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs, targets, track_instances: Instances):
        return self.dino(memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs, targets, track_instances)
    
    def _post_process_single_image(self, frame_res, track_instances, is_last):
        track_instances = Instances.cat([self._generate_empty_tracks(), track_instances])
        
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                prob = frame_res['pred_logits'][0, :].sigmoid()
                track_scores, labels = prob.max(-1)
        
        if self.training:
            pad_size = frame_res['dn_meta']['pad_size']
            track_instances.output_embedding = frame_res['hs_last_layer'][0, pad_size:]
        else:
            track_instances.output_embedding = frame_res['hs_last_layer'][0]

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        

        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
        
        if not is_last:
            out_track_instances = self.track_embed(track_instances, frame_res)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        
        return frame_res

    @staticmethod
    def get_frame_encoder_res(input_list, frame_index):
        res = [data[frame_index:frame_index+1] for data in input_list]
        return res
    
    @torch.no_grad()
    def inference_images(self, img, targets, track_instances=None):
        if track_instances is None:
            track_instances = self._generate_empty_tracks(num_queries=0)
        
        memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs = self.dino.forward_encoder(img)

        res = {'pred': [], 'track_instances': None}

        for frame_index in range(img.shape['tensors.shape'][0]):
            self.dino.timer_decoder.resume()
            
            if targets[frame_index]['frame_id'].item() == 1:
                track_instances = self._generate_empty_tracks(num_queries=0)
            self.track_base.update_record(targets[frame_index])
            frame_memory, frame_mask_flatten, frame_lvl_pos_embed_flatten, frame_valid_ratios = self.get_frame_encoder_res([memory, mask_flatten, lvl_pos_embed_flatten, valid_ratios], frame_index)
            frame_res = self._forward_single_image(frame_memory, frame_mask_flatten, spatial_shapes, frame_lvl_pos_embed_flatten, level_start_index, frame_valid_ratios, 1, None, track_instances)
            self.dino.timer_decoder.pause()
            self.dino.timer_tracking.resume()
            frame_res = self._post_process_single_image(frame_res, track_instances, False)
            track_instances = frame_res['track_instances']
            track_instances = self.post_process(track_instances, targets[frame_index]["orig_size"])
            res['pred'].append(track_instances.to(torch.device('cpu')))
            res['track_instances'] = track_instances
            self.dino.timer_tracking.pause()
        
        return res
    
    def forward(self, frames, targets=None, input_instances=None):
        # track_instances = []
        # frame_res = self._forward_single_image(frames, targets, track_instances)

        # return frame_res

        if not self.training:
            return self.inference_images(frames, targets, input_instances)

        if self.training:
            self.criterion.initialize_for_single_clip(input_instances, targets, self.dino.transformer.tgt_embed.weight.device)
        
        memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs = self.dino.forward_encoder(frames)
        
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }

        track_instances = self._generate_empty_tracks(num_queries=0)
        # keys = list(track_instances._fields.keys())
        for frame_index, target in enumerate(targets):
            is_last = frame_index == len(targets) - 1

            frame_memory, frame_mask_flatten, frame_lvl_pos_embed_flatten, frame_valid_ratios = self.get_frame_encoder_res([memory, mask_flatten, lvl_pos_embed_flatten, valid_ratios], frame_index)
            frame_res = self._forward_single_image(frame_memory, frame_mask_flatten, spatial_shapes, frame_lvl_pos_embed_flatten, level_start_index, frame_valid_ratios, 1, [target], track_instances)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)
            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
        
        # if not self.training:
        #     outputs['track_instances'] = track_instances
        return outputs
