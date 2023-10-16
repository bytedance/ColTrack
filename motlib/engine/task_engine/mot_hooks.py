# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import logging
import torch
import json
from pathlib import Path
from motlib.engine.utils.hooks import HookBase
from util.utils import ModelEma
import util.misc as utils
from .default_hooks import ResumeHook


class MotResumeHook(ResumeHook):
    def load_pretrain(self):
        logger = logging.getLogger(__name__)
        checkpoint = torch.load(self.trainer.args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = self.trainer.args.finetune_ignore if self.trainer.args.finetune_ignore else []
        ignorelist = []

        dino_flag = [False]

        def check_keep(keyname, ignorekeywordlist):
            if 'dino' in keyname:
                dino_flag[0] = True
            for keyword in ignorekeywordlist:
                if '/' not in keyword:
                    if keyword in keyname:
                        ignorelist.append(keyname)
                        return False
                else:
                    keyword = keyword.split('/')
                    all_match = True
                    for kw in keyword:
                        if kw not in keyname:
                            all_match = False
                            break
                    if all_match:
                        ignorelist.append(keyname)
                        return False

            return True

        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))

        if not dino_flag[0]:
            _load_output = self.trainer.model_without_ddp.dino.load_state_dict(_tmp_st, strict=False)
        else:
            _load_output = self.trainer.model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if self.use_ema:
            if 'ema_model' in checkpoint:
                self.trainer.ema_m.module.dino.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                raise NotImplementedError