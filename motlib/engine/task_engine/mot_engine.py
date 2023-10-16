# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from . import ENGIN_REGISTRY
from .train_loop import TrainerBase
from util.utils import ModelEma, BestMetricHolder
from motlib.utils import get_param_dict, task_setup, VisStorage
from motlib.mot_dataset.dataloader import get_dataloader
from datasets import get_coco_api_from_dataset
from motlib.evaluation.evaluators import run_evaluator
from . import default_hooks as hooks
from . import mot_hooks
import util.misc as utils
from . import SimpleTrainer
from motlib.mot_dataset.preprocessing import data_preprocess_e2e
from motlib.mot_models.network import build_network
from util.get_param_dicts import match_name_keywords


import torch
import logging
import json
import os
import math
import sys
from pathlib import Path


__all__ = ['MotSimpleTrainer']


@ENGIN_REGISTRY.register()
class MotSimpleTrainer(SimpleTrainer):
    def init(self):
        self.best_metric_key = 'MOTA'
        
    def frozen_model(self, model, init_flag=False):
        try:
            frozen_weights = self.args.frozen_weights_mot
        except:
            frozen_weights = []
        model.train()

        for n, m in model.named_modules():
            if match_name_keywords(n, frozen_weights):
                if init_flag:
                    self.logger.info('set model eval: {}'.format(n))
                m.eval()
                # m.requires_grad_(False)
        if init_flag:
            for n, p in model.named_parameters():
                if match_name_keywords(n, frozen_weights):
                    self.logger.info('frozen model: {}'.format(n))
                    p.requires_grad = False


    def run_step(self):
        samples, targets, img_info, img_id  = next(self.traindataloader_iterator)
        samples, targets, gt_instances = data_preprocess_e2e(self.device, samples, targets, img_info, img_id)

        with torch.cuda.amp.autocast(enabled=self.args.amp):
            if self.need_tgt_for_training:
                outputs = self.model(samples, targets, gt_instances)
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
        self.data_loader_train.dataset._dataset.step_epoch()
        return super().after_epoch()

    def build_hooks(self):
        res = super().build_hooks()
        res[0] = mot_hooks.MotResumeHook(self.args.use_ema, self.output_dir / 'model', self.args.resume, self.args.pretrain_model_path)
        return res

    def before_train(self):
        super().before_train()
        self.data_loader_train.dataset._dataset.set_epoch(self.epoch)

    def build_model(self):
        model, criterion, postprocessors = build_network('build_dinomot', self.args)

        wo_class_error = False
        model.to(self.device)
        self.frozen_model(model, init_flag=True)

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