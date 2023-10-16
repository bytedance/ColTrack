import json
import random
import time
from pathlib import Path
import os, sys

from util.logger import setup_logger
import numpy as np
import torch
import util.misc as utils
from util.slconfig import SLConfig
from motlib.utils import time_tag


__all__ = ['task_setup']


def task_setup(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    config_file = args.options['config_file']

    print("Loading config file from {}".format(config_file))
    time.sleep(args.rank * 0.02)
    
    cfg = SLConfig.fromfile(config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    _output_dir = Path(args.output_dir)
    if not _output_dir.is_absolute():
        _cur_dir = Path(os.path.realpath(__file__)).parent.parent.parent
        _output_dir = _cur_dir / args.output_dir
        args.output_dir = str(_output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    _job_name = str(_output_dir.name)
    args.job_name = _job_name

    if args.rank == 0:
        save_cfg_path = _output_dir / 'config'/ "config_cfg.py"
        save_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(save_cfg_path))
        save_json_path = _output_dir / 'config'/ "config_args_raw.json"
        with open(str(save_json_path), 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="motlib")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = _output_dir / 'config'/ "config_args_all.json"
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_json_path), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(str(save_json_path)))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')
    
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if hasattr(args, 'torch_home') and args.torch_home is not None:
        os.environ['TORCH_HOME'] = args.torch_home