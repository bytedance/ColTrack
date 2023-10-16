# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import copy
from typing import Optional, List
import math
import random
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from motlib.mot_models.structures import Instances, Boxes, pairwise_iou
from util import box_ops
from .qim_base import QueryInteractionBase, random_drop_tracks
from util.misc import inverse_sigmoid
from .motr import QueryInteractionDefault

from . import QIM_REGISTRY

__all__ = ['TimeTrackQIM']


@QIM_REGISTRY.register()
class TimeTrackQIM(QueryInteractionDefault):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.momentum = 0.9
        self.fp_history_ratio = self.args.fp_history_ratio
    
    def _add_fp_history(self, active_track_instances: Instances) -> Instances:
        if len(active_track_instances) <= 1:
            return active_track_instances
        
        boxes = Boxes(box_ops.box_cxcywh_to_xyxy(active_track_instances.pred_boxes))
        ious = pairwise_iou(boxes, boxes)
        ious = ious - torch.diag_embed(torch.diag(ious))
        fp_indexes = ious.max(dim=0).indices
        fp_embeding = active_track_instances.query_pos[fp_indexes].clone().detach()

        fp_fea_mask = torch.zeros_like(active_track_instances.scores).unsqueeze(-1) 

        for i in range(len(active_track_instances)):
            if len(active_track_instances.mem_bank[i]) >= 3 and random.uniform(0, 1) < self.fp_history_ratio:
                active_track_instances.mem_bank[i][-1] = fp_embeding[i]
                fp_fea_mask[i] = 1

        active_track_instances.query_pos = active_track_instances.query_pos * (1 - fp_fea_mask) + fp_fea_mask * fp_embeding

        return active_track_instances
    
    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        
        if self.add_kalmanfilter:
            track_instances.ref_pts = inverse_sigmoid(track_instances.predict_box())
        else:
            track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes.detach().clone())
        
        track_instances.query_pos = out_embed

        ref_box = track_instances.pred_boxes.detach().clone()

        for i in range(len(track_instances)):
            track_instances.mem_bank[i].append(out_embed[i])
            track_instances.ref_bank[i].append(ref_box[i])

        track_instances.track_time[:] += 1

        return track_instances
        

    def forward(self, data, frame_res) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        if self.training and self.fp_history_ratio>0:
            active_track_instances = self._add_fp_history(active_track_instances)
        return active_track_instances 