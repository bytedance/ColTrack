# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from models.dino.dino import SetCriterion
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

__all__ = ['MotSetCriterion']


@CRITERION_REGISTRY.register()
class MotSetCriterion(SetCriterion):
    def __init__(self, args, num_classes, matcher, weight_dict, focal_alpha, losses):
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
        self.args = args

        self._current_frame_idx = 0
        self.losses_dict = []
    
    def _step(self):
        self._current_frame_idx += 1
    
    def initialize_for_single_clip(self, gt_instances: List[Instances], targets, device):
        self.gt_instances = gt_instances
        self.targets = targets
        self.num_samples = 0
        self.sample_device = device
        self._current_frame_idx = 0
        self.losses_dict = []

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        self.num_boxes_for_mot = num_boxes
    
    def inherit_and_update_previous_tracks(self, track_instances, obj_idx_to_gt_idx):
        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(self.sample_device)
        matched_track_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(self.sample_device)
        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]
        return num_disappear_track, prev_matched_indices, unmatched_track_idxes
    
    def match_for_single_decoder_layer(self, unmatched_outputs, unmatched_track_idxes, untracked_gt_instances, untracked_tgt_indexes, matcher):
        new_track_indices = matcher(unmatched_outputs,
                                             [{"boxes":untracked_gt_instances.boxes, "labels":untracked_gt_instances.labels}])  # list[tuple(src_idx, tgt_idx)]

        src_idx = new_track_indices[0][0]
        tgt_idx = new_track_indices[0][1]
        # concat src and tgt.
        new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                            dim=1).to(self.sample_device)
        return new_matched_indices
    
    def get_untracked_gt_instances(self, track_instances, gt_instances_i):
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(self.sample_device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(self.sample_device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]
        return untracked_gt_instances, untracked_tgt_indexes
    
    def new_match_and_update(self, unmatched_outputs, unmatched_track_idxes, untracked_gt_instances, untracked_tgt_indexes, track_instances, gt_instances_i):
        # new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)
        
        new_matched_indices = self.match_for_single_decoder_layer(unmatched_outputs, unmatched_track_idxes, untracked_gt_instances, untracked_tgt_indexes, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))
        return new_matched_indices, track_instances
    
    def other_loss(self, outputs: dict):
        pass

    def match_for_single_frame(self, outputs: dict):
        self.other_loss(outputs)
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

        loss = self.cal_loss(outputs, [targets_i], matched_indices, unmatched_track_idxes, prev_matched_indices, untracked_gt_instances, untracked_tgt_indexes)
        self.losses_dict.append(loss)
        self._step()
        return track_instances
    
    def cal_loss(self, outputs, targets, matched_indices, unmatched_track_idxes, prev_matched_indices, untracked_gt_instances, untracked_tgt_indexes, return_indices=False):
        losses = {}

        matched_indices = [(matched_indices[:, 0], matched_indices[:, 1])]
        if return_indices:
            indices0_copy = matched_indices
            indices_list = []

        num_boxes = self.num_boxes_for_mot
        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))


            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)


        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, matched_indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = self.match_for_single_decoder_layer(unmatched_outputs_layer, unmatched_track_idxes, untracked_gt_instances, untracked_tgt_indexes, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                matched_indices_layer = [(matched_indices_layer[:, 0], matched_indices_layer[:, 1])]

                if return_indices:
                    indices_list.append(matched_indices_layer)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, matched_indices_layer, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']

            # unmatched_outputs_layer = {
            #         'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
            #         'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
            #     }
            # unmatched_track_idxes = torch.arange(interm_outputs['pred_logits'].shape[1], dtype=torch.long).to(self.sample_device)
            # indices = self.match_for_single_decoder_layer(interm_outputs, unmatched_track_idxes, untracked_gt_instances, untracked_tgt_indexes, self.matcher)
            # indices = [(indices[:, 0], indices[:, 1])]

            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                # if len(untracked_gt_instances) > 0:
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            raise NotImplementedError

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    
    def forward(self, outputs, targets, return_indices=False):
        loss_final = self.losses_dict[0]

        for frame_i in range(1, len(self.losses_dict)):
            loss_frame = self.losses_dict[frame_i]
            for k, v in loss_final.items():
                if k in loss_frame:
                    loss_final[k] = loss_frame[k] + v
            
            for k, v in loss_frame.items():
                if k not in loss_final:
                    loss_final[k] = v
        return loss_final