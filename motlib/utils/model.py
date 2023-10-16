import json
import torch
import torch.nn as nn
from util.get_param_dicts import match_name_keywords
from util.get_param_dicts import get_param_dict as get_param_dict_default


__all__ = ['get_param_dict']


def get_param_dict(args, model_without_ddp: nn.Module):
    try:
        param_dict_type = args.param_dict_type
    except:
        param_dict_type = 'default'
    assert param_dict_type in ['default', 'ddetr_in_mmdet', 'large_wd', 'finetune']
    if param_dict_type == 'finetune':
        ft_ignore_param = args.frozen_weights_mot 
        param_dicts = [
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad],
                "lr": args.lr
                }
        ]
    else:
        param_dicts = get_param_dict_default(args, model_without_ddp)
    return param_dicts