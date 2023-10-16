# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from pathlib import Path
from motlib.utils import set_dir
from . import trackeval
from .config_process import config_process



def evaluate_hota(args, dataset_name, eval_config, ts_path=None):

    evaluator, dataset_list, metrics_list = config_process(args, dataset_name, eval_config, ts_path=ts_path)

    return evaluator.evaluate(dataset_list, metrics_list)