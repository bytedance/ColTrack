# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from . import ENGIN_REGISTRY
from .train_loop import TrainerBase
from util.utils import ModelEma, BestMetricHolder
from motlib.utils import get_param_dict
from motlib.mot_dataset.dataloader import get_dataloader
from datasets import get_coco_api_from_dataset
from motlib.evaluation.evaluators import run_evaluator
from . import default_hooks as hooks
from motlib.mot_dataset.preprocessing.mot_tbd import data_preprocess_tbd
import util.misc as utils



import torch
import logging
import json
import os
import math
import sys
from pathlib import Path


__all__ = ['SimpleTrainer']




@ENGIN_REGISTRY.register()
class SimpleTrainer(TrainerBase):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)

        self.init()

        self.output_dir = Path(self.args.output_dir)
        self.logger = logging.getLogger(__name__)
        if not hasattr(args, 'just_inference_flag'):
            args.just_inference_flag = False
        if args.just_inference_flag:
            assert hasattr(self.args, 'num_classes')
        else:
            self.data_loader_train = self.build_train_dataloader()
            self.args.num_classes = self.data_loader_train.dataset.num_classes
            
        self.data_loader_val, self.base_ds = self.build_test_dataloader()
        
        self.args.dn_labelbook_size = self.args.num_classes 
        self.logger.info('num_classes {} dn_labelbook_size {}'.format(self.args.num_classes, self.args.dn_labelbook_size))
        model, model_without_ddp, criterion, postprocessors, ema_m, wo_class_error = self.build_model()
        optimizer = self.build_optimizer(model_without_ddp)
        self.lr_scheduler = self.build_scheduler(optimizer)

        self.model = model
        self.model_without_ddp = model_without_ddp
        self.optimizer = optimizer
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.ema_m = ema_m
        self.wo_class_error = wo_class_error

        self.best_metric_holder = BestMetricHolder(use_ema=args.use_ema)
        self.register_hooks(self.build_hooks())
    
    def init(self):
        self.best_metric_key = 'Det'
    
    def test(self):
        for h in self._hooks:
            if isinstance(h, hooks.ResumeHook):
                h.resume_or_pretrain()
                break
        for h in self._hooks:
            if isinstance(h, hooks.EvalHook):
                h.eval_func()
                break
    
    def test_all(self, model_path, begin_epoch=20):
        logger = logging.getLogger(__name__)
        model_dirs = Path(model_path).glob("checkpoint00*.pth")
        for model_dir in sorted(list(model_dirs)):
            self.epoch = int((model_dir.stem).split('int')[1])
            if self.epoch <= begin_epoch:
                logger.info('Ignore model {}'.format(str(model_dir.stem)))
                continue
            checkpoint = torch.load(str(model_dir), map_location='cpu')
            self.model_without_ddp.load_state_dict(checkpoint['model'])
            logger.info('resume model from {}!'.format(str(model_dir)))
            for h in self._hooks:
                if isinstance(h, hooks.EvalHook):
                    h.eval_func()
                    break

    def before_epoch(self):
        self.epoch_flag = True
        if self.args.distributed:
            self.data_loader_train.batch_sampler.sampler.set_epoch(self.epoch)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)

        try:
            need_tgt_for_training = self.args.use_dn
        except:
            need_tgt_for_training = False
        self.need_tgt_for_training = need_tgt_for_training
        self.frozen_model(self.model)
        self.criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        if not self.wo_class_error:
            metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(self.epoch)
        print_freq = 10
        self.iter_per_epoch = 0
        self.metric_logger = metric_logger
        self.iters_total_per_epoch = len(self.data_loader_train)

        if self.epoch >= self.args.aug_step:
            self.data_loader_train.close_mosaic()
        
        self.traindataloader_iterator = iter(metric_logger.log_every(self.data_loader_train, print_freq, header, logger=self.logger))
        super().before_epoch()
    
    def run_step(self):
        samples, targets, img_info, img_id  = next(self.traindataloader_iterator)
        samples, targets = data_preprocess_tbd(samples, targets, img_info, img_id)

        samples = samples.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=self.args.amp):
            if self.need_tgt_for_training:
                outputs = self.model(samples, targets)
            else:
                outputs = self.model(samples)
        
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            # import ipdb; ipdb.set_trace()
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            self.logger.error("Loss is {}, stopping training".format(loss_value))
            self.logger.error(loss_dict_reduced)
            sys.exit(1)
        
        max_norm = self.args.clip_max_norm
        # amp backward function
        if self.args.amp:
            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            if max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # original backward function
            self.optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()

        if self.args.onecyclelr:
            self.lr_scheduler.step()
        if self.args.use_ema:
            if self.epoch >= self.args.ema_epoch:
                self.ema_m.update(self.model)

        self.metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        if 'class_error' in loss_dict_reduced:
            self.metric_logger.update(class_error=loss_dict_reduced['class_error'])
        self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
    
    def after_epoch(self):
        if self.epoch + 1 >= self.epochs_total:
            self.train_flag = False
        if not self.args.onecyclelr:
            self.lr_scheduler.step()
        super().after_epoch()
    
    def after_step(self):
        self.iter_per_epoch += 1

        if self.iter_per_epoch % self.args.resize_step == 0:
            self.data_loader_train.random_resize()

        if self.iter_per_epoch >= self.iters_total_per_epoch:
            self.epoch_flag = False
        return super().after_step()
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        resume_hook = hooks.ResumeHook(self.args.use_ema, self.output_dir / 'model', self.args.resume, self.args.pretrain_model_path)
        ret = [resume_hook, hooks.VisualHook(), hooks.PeriodicCheckpointer(self.output_dir / 'model', self.args.save_checkpoint_interval)]

        def test_and_save_results():
            res = run_evaluator(self.args.evaluator,
                self.model, self.criterion, self.postprocessors, self.data_loader_val, self.base_ds, self.device, self.args.output_dir,
                wo_class_error=self.wo_class_error, args=self.args)
            if self.args.just_inference_flag:
                return
            map_regular = res[self.best_metric_key]
            if hasattr(self, 'visualizer'):
                self.visualizer.plot('add_scalars', 'Performance', res, self.epoch)
            _isbest = self.best_metric_holder.update(map_regular, self.epoch, is_ema=False)
            if _isbest:
                checkpoint_path = self.output_dir / 'model' /'checkpoint_best_regular.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                utils.save_on_master({
                    'model': self.model_without_ddp.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch': self.epoch,
                    'args': self.args,
                }, checkpoint_path)
            
            if self.args.use_ema:
                res_ema = run_evaluator(self.args.evaluator,
                    self.ema_m.module, self.criterion, self.postprocessors, self.data_loader_val, self.base_ds, self.device, self.args.output_dir,
                    wo_class_error=self.wo_class_error, args=self.args)
                map_ema = res_ema[self.best_metric_key]
                _isbest = self.best_metric_holder.update(map_ema, self.epoch, is_ema=True)
                if _isbest:
                    checkpoint_path = self.output_dir / 'model' / 'checkpoint_best_ema.pth'
                    utils.save_on_master({
                        'model': self.ema_m.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict(),
                        'epoch': self.epoch,
                        'args': self.args,
                    }, checkpoint_path)
                res_plot = {}
                for k, v in res_ema.items():
                    res_plot[k+'ema'] = v
                if hasattr(self, 'visualizer'):
                    self.visualizer.plot('add_scalars', 'Performance', res_plot, self.epoch)
            return

        ret.append(hooks.EvalHook(self.args.test_step, test_and_save_results))
        ret.append(hooks.IterationTimer())
        return ret
    
    def build_train_dataloader(self):
        data_loader_train = get_dataloader(self.args.train_dataloader, self.args)
        return data_loader_train
    
    def build_test_dataloader(self):
        data_loader_val = get_dataloader(self.args.test_dataloader, self.args)
        base_ds = get_coco_api_from_dataset(data_loader_val.dataset)
        return data_loader_val, base_ds
    
    def build_scheduler(self, optimizer):
        if self.args.onecyclelr:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.data_loader_train), epochs=self.args.epochs, pct_start=0.2)
        elif self.args.multi_step_lr:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_drop_list)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop)
        return lr_scheduler
    
    def build_model(self):
        from models.registry import MODULE_BUILD_FUNCS
        assert self.args.modelname in MODULE_BUILD_FUNCS._module_dict
        build_func = MODULE_BUILD_FUNCS.get(self.args.modelname)
        model, criterion, postprocessors = build_func(self.args)

        wo_class_error = False
        model.to(self.device)

        if self.args.use_ema:
            ema_m = ModelEma(model, self.args.ema_decay)
        else:
            ema_m = None
        
        model_without_ddp = model
        if self.args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=self.args.find_unused_params)
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info('number of params:'+str(n_parameters))
        self.logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))
        return model, model_without_ddp, criterion, postprocessors, ema_m, wo_class_error
    
    def frozen_model(self, model, init_flag=False):
        model.train()
    
    def build_optimizer(self, model):
        param_dicts = get_param_dict(self.args, model)

        optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        return optimizer