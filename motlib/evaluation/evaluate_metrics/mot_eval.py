# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import argparse
import os
import random
import warnings
import glob
import logging
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


def compare_dataframes(gts, ts, logger):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def evaluate_mota(args, dataset_name, eval_config, ts_path=None):
    gt_path = eval_config['gt_path']
    logger = logging.getLogger(__name__)
    evaluate_track(args, dataset_name, gt_path, ts_path, logger)
    

def evaluate_track(args, dataset_name, gt_path, ts_path=None, logger=None):
    mm.lap.default_solver = 'lap'
    gt_path = Path(gt_path) / 'gt'
    if ts_path is None:
        ts_path = Path(args.output_dir) / args.tracker_name / "track_results"
    gtfiles = list(gt_path.glob("*.txt"))
    tsfiles = list(ts_path.glob("*.txt"))

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    gt = OrderedDict([(str(f.stem), mm.io.loadtxt(str(f), fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(str(f.stem), mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])  

    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts, logger)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    logger.info(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    logger.info(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')