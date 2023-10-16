# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.utils import Registry


CRITERION_REGISTRY = Registry("MOTCRITERION")

from .default_criterion import *
from .time_track import *

def build_criterion(name, *ar, **kwargs):
    return CRITERION_REGISTRY.get(name)(*ar, **kwargs)