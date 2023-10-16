import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from motlib.mot_models.structures import Instances, Boxes, pairwise_iou
from util import box_ops
from .qim_base import QueryInteractionBase, random_drop_tracks
from util.misc import inverse_sigmoid

from . import QIM_REGISTRY

__all__ = ['QueryInteractionDefault']


@QIM_REGISTRY.register()
class QueryInteractionDefault(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.add_kalmanfilter = args.add_kalmanfilter
        
    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        return

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]
                    fp_track_instances.obj_idxes = fp_track_instances.obj_idxes - 2

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, track_instances) -> Instances:
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances: # 更新了query的嵌入和位置嵌入，还有参考点
        if len(track_instances) == 0:
            return track_instances
        
        is_pos = track_instances.scores > -1
        
        out_embed = track_instances.output_embedding
        track_instances.query_pos[is_pos] = out_embed[is_pos]
        if self.add_kalmanfilter:
            track_instances.ref_pts = inverse_sigmoid(track_instances.predict_box())
        else:
            track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes.detach().clone())
        return track_instances

    def forward(self, data, frame_res) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        return active_track_instances 