# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). All Bytedance's Modifications are Copyright (2023) Bytedance Ltd. and/or its affiliates. 


from util.utils import slprint, to_device
from . import EVALUATOR_REGISTRY

import time
import torch
import logging
from copy import deepcopy
from pathlib import Path
import util.misc as utils
import torch.distributed as dist
from motlib.evaluation.evaluate_metrics.cocoeval import CocoEvaluator
from motlib.evaluation.evaluate_metrics import mot_eval_metrics
from motlib.utils import set_dir, time_synchronized
from motlib.evaluation.utils.result_process import filter_result
from motlib.mot_models.network.dino_mot.tracker.manager import E2ETrackManager
from motlib.tracker.interpolation import GSInterpolation as iptrack


__all__ = ['evaluate_e2e']


@EVALUATOR_REGISTRY.register()
@torch.no_grad()
def evaluate_e2e(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None):
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    savefolder = output_dir / 'results'

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    if args.save_results:
        if utils.get_rank() in [0, -1]:
            set_dir(str(savefolder))
        dist.barrier()

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        logger.info("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    _cnt = 0
    output_state_dict = {'target': [], 'result': []} # for debug only

    track_instances = None
    valid_frame_num = data_loader.sampler.valid_num
    track_mannager = E2ETrackManager(args, valid_frame_num, deepcopy(data_loader.dataset.coco.dataset['videos']))

    inference_time = 0
    track_time = 0
    n_samples = 0
    warmup_iter = min(100, len(data_loader)//3)
    
    for cur_iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header, logger=logger)):
        samples = samples.to(device)
        # targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        is_time_record = (cur_iter < len(data_loader) - 1) and cur_iter > warmup_iter
        if is_time_record:
            n_samples += len(targets)
            start = time.time()

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, targets, track_instances)
            track_instances = outputs['track_instances']
        
        if is_time_record:
            infer_end = time_synchronized()
            inference_time += infer_end - start
        
        results = track_mannager.update(outputs['pred'], targets)

        if is_time_record:
            track_end = time_synchronized()
            track_time += track_end - infer_end

        _cnt += len(results)

        if len(results) > 0:

            if coco_evaluator is not None:
                coco_results = {}
                for img_id, det_res in results:
                    coco_results[img_id] = det_res
                coco_evaluator.update(coco_results)
            if args.save_results:
                for i, (img_id, res) in enumerate(results):
                    tgt = targets[i]
                    assert tgt['image_id'].item() == img_id
                    _image_id = tgt['image_id']
                    _frame_id = tgt['frame_id']
                    _video_id = tgt['video_id']
                    _res_box = res['boxes']
                    _res_score = res['scores']
                    _res_label = res['labels']

                    gt_info = torch.cat((_image_id, _frame_id, _video_id), 0)
                    
                    res_info = torch.cat((_res_box, _res_score.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                    # import ipdb;ipdb.set_trace()

                    output_state_dict['target'].append(gt_info.cpu())
                    output_state_dict['result'].append(res_info.cpu())
        del targets, samples, results
    
    
    resnet_forward_time = model.module.dino.timer_resnet.avg_seconds()
    encoder_forward_time = model.module.dino.timer_encoder.avg_seconds()
    decoder_forward_time = model.module.dino.timer_decoder.avg_seconds()
    tracking_forward_time = model.module.dino.timer_tracking.avg_seconds()
    
    statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples, resnet_forward_time, encoder_forward_time, decoder_forward_time, tracking_forward_time])
    if utils.is_dist_avail_and_initialized():
        torch.distributed.reduce(statistics, dst=0)
    
    inference_time = statistics[0].item()
    track_time = statistics[1].item()
    n_samples = statistics[2].item()

    resnet_forward_time = statistics[3].item()
    encoder_forward_time = statistics[4].item()
    decoder_forward_time = statistics[5].item()
    tracking_forward_time = statistics[6].item()

    a_infer_time = 1000 * inference_time / n_samples
    a_track_time = 1000 * track_time / n_samples

    time_info = ", ".join(
        [
            "Average {} time: {:.2f} ms".format(k, v)
            for k, v in zip(
                ["forward", "track", "inference"],
                [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
            )
        ]
    )
    time_info += ' fps {:.2f}'.format(1000 / (a_infer_time + a_track_time))
    logger.info(time_info)

    logger.info('Timer Resnet {:.2f} Encoder {:.2f} Decoder {:.2f} Track update {:.2f} Total {:.2f}'.format(resnet_forward_time, encoder_forward_time, decoder_forward_time, tracking_forward_time, inference_time))

    if args.save_results:
        savepath = savefolder/ 'results-{}.pkl'.format(utils.get_rank())
        savepath = str(savepath)
        logger.info("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(str(metric_logger)))
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    dist.barrier()
    res_record = torch.FloatTensor(5).cuda()
    res_record_dti = torch.FloatTensor(5).cuda()
    if utils.get_rank() in [-1, 0]:

        track_results_path = '{}/{}/track_results'.format(args.output_dir, args.tracker_name)
        interpolation_track_results_path = '{}/{}/track_results'.format(args.output_dir, 'IPTrack')
        set_dir(interpolation_track_results_path)
        iptrack(txt_path=track_results_path, save_path=interpolation_track_results_path)

        if not args.just_inference_flag:
            dataset_name = data_loader.dataset.dataset_name
            assert len(dataset_name) == 1
            output_res, _ = mot_eval_metrics(args, dataset_name=dataset_name[0], eval_config=data_loader.dataset.coco.eval_config)
            res_record = filter_result(stats, output_res, args.tracker_name, data_loader.dataset.coco.eval_config['dataset_class'], res_record)
            res_record_dti = filter_result(stats, output_res, 'IPTrack', data_loader.dataset.coco.eval_config['dataset_class'], res_record_dti)
    if utils.is_dist_avail_and_initialized():
            dist.barrier()
            dist.broadcast(res_record, 0)
            dist.broadcast(res_record_dti, 0)
    output = {'Det':res_record[0].item(),'HOTA': res_record[1].item(), 'MOTA': res_record[2].item(), 'IdSW': res_record[3].item(), 'IDF1': res_record[4].item(), 'HOTA_dti': res_record_dti[1].item(), 'MOTA_dti': res_record_dti[2].item(), 'IdSW_dti': res_record_dti[3].item(), 'IDF1_dti': res_record_dti[4].item()}
    return output
