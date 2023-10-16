from models.dino.matcher import HungarianMatcher

import torch, os
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from . import MOTMATCHER_REGISTRY


__all__ = ['MotHungarianMatcher']


@MOTMATCHER_REGISTRY.register()
class MotHungarianMatcher(HungarianMatcher):
    pass