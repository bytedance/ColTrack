# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 


from util.utils import slprint, to_device
from . import EVALUATOR_REGISTRY

import torch
import time
import logging
from copy import deepcopy
from pathlib import Path
import util.misc as utils
import torch.distributed as dist
from motlib.evaluation.evaluate_metrics.cocoeval import CocoEvaluator
from motlib.tracker.offline_track import track_by_offline
from motlib.evaluation.evaluate_metrics import mot_eval_metrics
from motlib.utils import torch_distributed_zero_first, set_dir, load_dino_results, time_synchronized
from motlib.evaluation.utils.result_process import filter_result
# from motlib.tracker.interpolation import dti as iptrack
from motlib.tracker.interpolation import GSInterpolation as iptrack


__all__ = ['my_evaluate_coco']


@EVALUATOR_REGISTRY.register()
@torch.no_grad()
def my_evaluate_coco(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None):
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    savefolder = output_dir / 'results'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        logger.info("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)

    if not args.ignore_det:
        try:
            need_tgt_for_training = args.use_dn
        except:
            need_tgt_for_training = False

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

        inference_time = 0
        n_samples = 0
        warmup_iter = min(100, len(data_loader) // 3)

        _cnt = 0
        output_state_dict = {'target': [], 'result': []} # for debug only
        for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
            samples = samples.to(device)
            targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

            is_time_record = _cnt < len(data_loader) - 1 and _cnt > warmup_iter
            if is_time_record:
                n_samples += len(targets)
                start = time.time()

            with torch.cuda.amp.autocast(enabled=args.amp):
                if need_tgt_for_training:
                    outputs = model(samples, targets)
                else:
                    outputs = model(samples)
                
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
            if 'class_error' in loss_dict_reduced:
                metric_logger.update(class_error=loss_dict_reduced['class_error'])

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            # [scores: [100], labels: [100], boxes: [100, 4]] x B
        
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            # import ipdb; ipdb.set_trace()
            if coco_evaluator is not None:
                coco_evaluator.update(res)
            if args.save_results:
                for i, (tgt, res) in enumerate(zip(targets, results)):

                    _image_id = tgt['image_id']
                    _frame_id = tgt['frame_id']
                    _video_id = tgt['video_id']
                    _res_box = res['boxes']
                    _res_score = res['scores']
                    _res_label = res['labels']
                    if data_loader.dataset.coco.eval_config['name'] == 'Default':
                        select_idx = _res_label <= 1
                        _res_box = _res_box[select_idx]
                        _res_score = _res_score[select_idx]
                        _res_label = _res_label[select_idx]

                    gt_info = torch.cat((_image_id, _frame_id, _video_id), 0)
                    
                    res_info = torch.cat((_res_box, _res_score.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                    # import ipdb;ipdb.set_trace()

                    output_state_dict['target'].append(gt_info.cpu())
                    output_state_dict['result'].append(res_info.cpu())
            del targets, samples, res, results

            _cnt += 1
            if args.debug:
                if _cnt % 15 == 0:
                    print("BREAK!"*5)
                    break

        statistics = torch.cuda.FloatTensor([inference_time, n_samples])
        if utils.is_dist_avail_and_initialized():
            torch.distributed.reduce(statistics, dst=0)
        
        inference_time = statistics[0].item()
        n_samples = statistics[1].item()

        a_infer_time = 1000 * inference_time / n_samples

        time_info = ", ".join(
            [
                "Average formard time: {:.2f} ms fps {:.2f}".format(a_infer_time, 1000 / a_infer_time)
            ]
        )
        logger.info(time_info)

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
    else:
        stats = None
        # stats = {}
        # det_result = load_dino_results(str(savefolder))
        # for k, v in det_result.items():
        #     _, det_res = v
        #     res = {'boxes': det_res[:, :4], 'scores': det_res[:, 4], 'labels': det_res[:, 5]}
        #     if coco_evaluator is not None:
        #         coco_evaluator.update({k: res})
        # coco_evaluator.synchronize_between_processes()
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        # if 'bbox' in postprocessors.keys():
        #     stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            
        # if 'segm' in postprocessors.keys():
        #     stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    dist.barrier()
    res_record = torch.FloatTensor(5).cuda()
    res_record_dti = torch.FloatTensor(5).cuda()
    if utils.get_rank() in [-1, 0]:
        track_by_offline(args, deepcopy(data_loader.dataset.coco.dataset['videos']), str(savefolder), len(data_loader.dataset))
        track_results_path = '{}/{}/track_results'.format(args.output_dir, args.tracker_name)
        interpolation_track_results_path = '{}/{}/track_results'.format(args.output_dir, 'IPTrack')
        set_dir(interpolation_track_results_path)
        iptrack(txt_path=track_results_path, save_path=interpolation_track_results_path)
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
