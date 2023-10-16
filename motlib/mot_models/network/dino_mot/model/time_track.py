# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from . import MOTMODEL_REGISTRY

from .base import MOTDINOBASE
import torch
import copy
from collections import deque
from torch import nn
from motlib.mot_models.structures import Instances
from motlib.mot_models.network.dino_mot.tracker.util import RuntimeTrackerBase, TrackerPostProcess
from .default_model import MOTDINO

__all__ = ['TimeTrackDINO']


@MOTMODEL_REGISTRY.register()
class TimeTrackDINO(MOTDINO):
    
    def _post_process_single_image(self, frame_res, track_instances, is_last):
        track_instances = Instances.cat([self._generate_empty_tracks(), track_instances])

        decoder_out = frame_res['track_dec_out']
        pad_size = decoder_out.get('pad_size', 0)
        det_query_num = decoder_out.get('det_query_num', 0)
        track_query_num = decoder_out.get('track_query_num', 0)


        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                prob = frame_res['pred_logits'][0, :].sigmoid()
                track_scores, labels = prob.max(-1)
        
        if self.training:
            pad_size = frame_res['dn_meta']['pad_size']
            track_instances.output_embedding = frame_res['hs_last_layer'][0, pad_size:pad_size+det_query_num+track_query_num]
        else:
            track_instances.output_embedding = frame_res['hs_last_layer'][0, :pad_size+det_query_num+track_query_num]

        track_instances.scores = track_scores[:det_query_num+track_query_num]
        track_instances.pred_logits = frame_res['pred_logits'][0, :det_query_num+track_query_num]
        track_instances.pred_boxes = frame_res['pred_boxes'][0, :det_query_num+track_query_num]

        if track_scores.shape[0] > det_query_num+track_query_num:
            frame_res['track_dec_out']['decoder_out']['time_tracker_pred'] = {}

            frame_res['track_dec_out']['decoder_out']['time_tracker_pred'][5] = {
                'pred_logits': frame_res['pred_logits'][:, det_query_num+track_query_num:], 
                'pred_boxes': frame_res['pred_boxes'][:, det_query_num+track_query_num:]
                }
            new_aux_outputs = []
            for layer_i, layer_aux_output in enumerate(frame_res['aux_outputs']):
                frame_res['track_dec_out']['decoder_out']['time_tracker_pred'][layer_i] = {
                'pred_logits': layer_aux_output['pred_logits'][:, det_query_num+track_query_num:], 
                'pred_boxes': layer_aux_output['pred_boxes'][:, det_query_num+track_query_num:]
                }
                new_aux_outputs.append({
                'pred_logits': layer_aux_output['pred_logits'][:, :det_query_num+track_query_num], 
                'pred_boxes': layer_aux_output['pred_boxes'][:, :det_query_num+track_query_num]
                })
        
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :det_query_num+track_query_num]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :det_query_num+track_query_num]
            frame_res['aux_outputs'] = new_aux_outputs
        
    
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