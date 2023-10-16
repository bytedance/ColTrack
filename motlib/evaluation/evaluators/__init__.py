# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.utils import Registry


EVALUATOR_REGISTRY = Registry("EVALUATOR")

from .coco import *
from .mot_e2e import *
from .my_coco import *

def run_evaluator(name, *args, **kwargs):
    return EVALUATOR_REGISTRY.get(name)(*args, **kwargs)