# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from . import MotSetCriterion
from motlib.mot_models.structures import Instances
from motlib.mot_models.structures.boxes import Boxes, matched_boxlist_iou
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from . import CRITERION_REGISTRY 

__all__ = ['TimeTrackMotSetCriterion']


@CRITERION_REGISTRY.register()
class TimeTrackMotSetCriterion(MotSetCriterion):
    def __init__(self, args, num_classes, matcher, weight_dict, focal_alpha, losses):
        super().__init__(args, num_classes, matcher, weight_dict, focal_alpha, losses)

        weight_dict = {
            'loss_ce_time':self.weight_dict['loss_ce'], 
            'loss_bbox_time':self.weight_dict['loss_bbox'], 
            'loss_giou_time':self.weight_dict['loss_giou']
            }
        
        for i in range(self.args.dec_layers):
            layer_weight_dict = {}
            for k, v in weight_dict.items():
                layer_weight_dict[k+'_{}'.format(i)] = v
            self.weight_dict.update(layer_weight_dict)

    def initialize_for_single_clip(self, gt_instances: List[Instances], targets, device):
        super().initialize_for_single_clip(gt_instances, targets, device)

        self.num_pad_box = 0
    
    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        targets_i = self.targets[self._current_frame_idx]
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.
        assert (pred_boxes_i[:, 2:] >= 0).all()
        assert (gt_instances_i.boxes[:, 2:] >= 0).all()
        device = pred_logits_i.device

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}

        # step1. inherit and update the previous tracks.
        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        num_disappear_track, prev_matched_indices, unmatched_track_idxes = self.inherit_and_update_previous_tracks(track_instances, obj_idx_to_gt_idx)
        loss = self.other_loss(outputs, prev_matched_indices)

        # step3. select the untracked gt instances (new tracks).
        untracked_gt_instances, untracked_tgt_indexes = self.get_untracked_gt_instances(track_instances, gt_instances_i)
        
        # match_for_single_decoder_layer = self.match_with_untracked_gt_instances(track_instances, gt_instances_i, unmatched_track_idxes)

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices, track_instances = self.new_match_and_update(unmatched_outputs, unmatched_track_idxes, untracked_gt_instances, untracked_tgt_indexes, track_instances, gt_instances_i)

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = device

        loss.update(self.cal_loss(outputs, [targets_i], matched_indices, unmatched_track_idxes, prev_matched_indices, untracked_gt_instances, untracked_tgt_indexes))
        self.losses_dict.append(loss)
        self._step()
        return track_instances

    def other_loss(self, outputs, prev_matched_indices):

        loss_name_tmp = ['loss_ce_time_5', 'class_error_time_5', 'loss_bbox_time_5', 'loss_giou_time_5', 'loss_xy_time_5', 'loss_hw_time_5', 'cardinality_error_time_5', 'loss_ce_time_0', 'class_error_time_0', 'loss_bbox_time_0', 'loss_giou_time_0', 'loss_xy_time_0', 'loss_hw_time_0', 'cardinality_error_time_0', 'loss_ce_time_1', 'class_error_time_1', 'loss_bbox_time_1', 'loss_giou_time_1', 'loss_xy_time_1', 'loss_hw_time_1', 'cardinality_error_time_1', 'loss_ce_time_2', 'class_error_time_2', 'loss_bbox_time_2', 'loss_giou_time_2', 'loss_xy_time_2', 'loss_hw_time_2', 'cardinality_error_time_2', 'loss_ce_time_3', 'class_error_time_3', 'loss_bbox_time_3', 'loss_giou_time_3', 'loss_xy_time_3', 'loss_hw_time_3', 'cardinality_error_time_3', 'loss_ce_time_4', 'class_error_time_4', 'loss_bbox_time_4', 'loss_giou_time_4', 'loss_xy_time_4', 'loss_hw_time_4', 'cardinality_error_time_4']

        track_dec_out = outputs['track_dec_out']
        track_num = track_dec_out.get('track_query_num', 0)
        if track_num == 0 or 'time_label' not in track_dec_out['decoder_out']:
            loss = {}
            for k in loss_name_tmp:
                loss[k] = torch.zeros((2), device=outputs['pred_logits'].device).mean()
            return loss

        loss = self.time_track_match_loss(track_num, copy.deepcopy(prev_matched_indices), track_dec_out)
        
        return loss
        
    def time_track_match_loss(self, track_num, prev_matched_indices, track_dec_out):
        track_match = torch.zeros((track_num, 2), dtype=prev_matched_indices.dtype, device=prev_matched_indices.device) - 1
        det_query_num = track_dec_out['det_query_num']
        time_tracker_label = track_dec_out['decoder_out']['time_label']
        prev_matched_indices[:, 0] = prev_matched_indices[:, 0] - det_query_num
        track_match[prev_matched_indices[:, 0], 1] = prev_matched_indices[:, 1]

        time_tracker_match = track_match[time_tracker_label]
        time_tracker_match[:, 0] = torch.arange(time_tracker_match.shape[0], dtype=torch.long).to(self.sample_device)
        time_tracker_match = time_tracker_match[time_tracker_match[:, 1]>=0]

        self.num_pad_box += prev_matched_indices.shape[0]
        
        matched_indices = [(time_tracker_match[:, 0], time_tracker_match[:, 1])]

        losses = {}
        for layer_id, v in track_dec_out['decoder_out']['time_tracker_pred'].items():
            for loss in self.losses:
                    # if len(untracked_gt_instances) > 0:
                    l_dict = self.get_loss(loss, v, [self.targets[self._current_frame_idx]], matched_indices, 1)
                    l_dict = {k + '_time_{}'.format(layer_id): v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
        return losses

    
    def get_time_track_boxes_num(self):
        num_pad_box = torch.as_tensor([self.num_pad_box], dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pad_box)
        num_pad_box = torch.clamp(num_pad_box / get_world_size(), min=1).item()
        return num_pad_box
    
    def forward(self, outputs, targets, return_indices=False):
        loss_final = super().forward(outputs, targets, return_indices)

        num_boxes = self.get_time_track_boxes_num()

        for k, v in loss_final.items():
            for new_name in ['loss_ce_time', 'loss_bbox_time', 'loss_giou_time']:
                if new_name in k:
                    loss_final[k] = v / num_boxes
        
        return loss_final