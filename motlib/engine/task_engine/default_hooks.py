# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import logging
import torch
import json
import time
import datetime
from pathlib import Path
from motlib.engine.utils.hooks import HookBase
from motlib.utils import Timer
from util.utils import ModelEma
import util.misc as utils
from motlib.utils import VisStorage


class EvalHook(HookBase):
    def __init__(self, eval_step, eval_func) -> None:
        self.eval_func = eval_func
        self.eval_step = eval_step

    def after_epoch(self):
        if ( self.trainer.epoch + 1 ) % self.eval_step == 0:
            try:
                self.eval_func()
            except:
                logger = logging.getLogger(__name__)
                logger.error('During test, error raised.')
    
    def after_train(self):
        if self.trainer.epoch % self.eval_step != 0:
            self.eval_func()

class VisualHook(HookBase):
    def __init__(self) -> None:
        pass

    def before_train(self):
        self.visualizer = VisStorage(self.trainer.args, self.trainer.epoch)
        self.trainer.visualizer = self.visualizer

    def after_epoch(self):
        loss_str = {}
        for name, meter in self.trainer.metric_logger.meters.items():
            loss_str[str(name)] = meter.median
        self.visualizer.plot('add_scalars', 'Loss', loss_str, self.trainer.epoch)


class PeriodicCheckpointer(HookBase):
    def __init__(self, outout_dir, save_step=1) -> None:
        self.output_dir = outout_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_step = save_step

    def after_epoch(self):
        epoch = self.trainer.epoch
        checkpoint_paths = [self.output_dir / 'checkpoint.pth']
        if (epoch + 1) % self.trainer.args.lr_drop == 0 or (epoch + 1) % self.save_step == 0:
            checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            weights = {
                'model': self.trainer.model_without_ddp.state_dict(),
                'optimizer': self.trainer.optimizer.state_dict(),
                'lr_scheduler': self.trainer.lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': self.trainer.args,
            }
            if self.trainer.args.use_ema:
                weights.update({
                    'ema_model': self.trainer.ema_m.module.state_dict(),
                })
            utils.save_on_master(weights, checkpoint_path)
        if self.trainer.args.distributed:
            torch.distributed.barrier()


class ResumeHook(HookBase):
    def __init__(self, use_ema, output_dir, resume=None, pretrain_model_path=None):
        self.use_ema = use_ema
        self.pretrain_model_path = pretrain_model_path
        self._file_dir = Path(output_dir) / 'checkpoint.pth'
        if self._file_dir.exists():
            self._resume = str(self._file_dir)
        else:
            self._resume = resume
    
    def before_train(self):
        self.resume_or_pretrain()
    
    def load_frozen_weights(self):
        logger = logging.getLogger(__name__)
        checkpoint = torch.load(self.trainer.args.frozen_weights, map_location='cpu')
        self.trainer.model_without_ddp.detr.load_state_dict(checkpoint['model'])
        logger.info("load frozen_weight from {}".format(self.trainer.args.frozen_weights))
    
    def resume(self):
        logger = logging.getLogger(__name__)
        if self._resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(self._resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(self._resume, map_location='cpu')
        self.trainer.model_without_ddp.load_state_dict(checkpoint['model'])
        logger.info('resume model from {}!'.format(self._resume))
        if self.use_ema:
            if 'ema_model' in checkpoint:
                self.trainer.ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del self.trainer.ema_m
                self.trainer.ema_m = ModelEma(self.trainer.model, self.trainer.args.ema_decay)                

        if not self.trainer.args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            self.trainer.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.trainer.args.start_epoch = checkpoint['epoch'] + 1
    
    def load_pretrain(self):
        logger = logging.getLogger(__name__)
        checkpoint = torch.load(self.trainer.args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = self.trainer.args.finetune_ignore if self.trainer.args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
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

        _load_output = self.trainer.model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if self.use_ema:
            if 'ema_model' in checkpoint:
                self.trainer.ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del self.trainer.ema_m
                self.trainer.ema_m = ModelEma(self.trainer.model, self.trainer.args.ema_decay) 
    
    def resume_or_pretrain(self):
        logger = logging.getLogger(__name__)
        if self.trainer.args.frozen_weights is not None:
            self.load_frozen_weights()

        if self._resume:
            self.resume()
        
        elif self.pretrain_model_path:
            self.load_pretrain()
        self.trainer.epoch = self.trainer.args.start_epoch
        if not self.trainer.args.eval:
            self.trainer.epochs_total = self.trainer.args.epochs
            self.trainer.start_iter = self.trainer.epoch * len(self.trainer.data_loader_train)
            self.trainer.iter = self.trainer.start_iter
            self.trainer.train_flag = True if self.trainer.epoch < self.trainer.epochs_total else False
            logger.info("Starting training from iteration {} epoch {}".format(self.trainer.iter, self.trainer.epoch))


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.
    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=5):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer = Timer()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )
        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()
