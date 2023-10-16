# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from ast import arg
from .mot_eval import evaluate_mota
from .trackeval_manager.manager import evaluate_hota


def mot_eval_metrics(arg, *args, **kwds):
    _eval_func_bank = {
        'CLEAR': evaluate_mota,
        'HOTA': evaluate_hota
    }
    mot_metrics = arg.mot_metrics
    if mot_metrics not in _eval_func_bank:
        raise KeyError
    return _eval_func_bank[mot_metrics](arg, *args, **kwds)
