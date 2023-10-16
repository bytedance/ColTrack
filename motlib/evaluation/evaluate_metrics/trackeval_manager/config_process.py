# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from pathlib import Path
import logging
import numpy as np
import json
from motlib.utils import set_dir
from motlib.mot_dataset.dataset_prepare.utils.coco_format import MotCOCO
from copy import deepcopy
from .dataset import get_eval_dataset_func
from . import trackeval
from .trackeval import metrics

def config_process(args, dataset_name, eval_config, ts_path=None):
    if eval_config['name'] == 'Default':
        return config_process_default(args, dataset_name, eval_config, ts_path)
    elif eval_config['name'] == 'BDD100K':
        return config_process_bdd100k(args, dataset_name, eval_config, ts_path)
    else:
        raise KeyError


def config_process_default(args, dataset_name, eval_config, ts_path=None):
    eval_config = deepcopy(eval_config)
    gt_path = eval_config['gt_path']
    gt_path = Path(gt_path)
    ts_path = args.output_dir if ts_path is None else ts_path
    output_folder = Path(ts_path) / args.tracker_name / 'eval'
    set_dir(output_folder)

    evaluator_config = eval_config['evaluator']
    evaluator_config['USE_PARALLEL'] = False
    evaluator_config['LOG_ON_ERROR'] = str(output_folder / 'error_log.txt')

    dataset_config = eval_config['dataset']

    dataset_config_new = {
        'BENCHMARK' : dataset_name,
        'SKIP_SPLIT_FOL' : True,
        "GT_FOLDER": str(gt_path),
        'GT_LOC_FORMAT': "{gt_folder}/gt/{seq}.txt",
        "SEQMAP_FILE": str(gt_path / 'seq_info.json'),
        'TRACKERS_FOLDER' : ts_path,
        'TRACKERS_TO_EVAL': [args.tracker_name, 'IPTrack'],
        'TRACKER_SUB_FOLDER': "track_results",
        "OUTPUT_FOLDER" : str(output_folder)
    }
    dataset_config.update(dataset_config_new)
    metrics_config = eval_config['metrics']

    evaluator = trackeval.Evaluator(evaluator_config)
    dataset_class_func = get_eval_dataset_func(eval_config['dataset_class'])
    dataset_list = [dataset_class_func(dataset_config)]

    metrics_list = []
    for mt in [metrics.HOTA, metrics.CLEAR, metrics.Identity, metrics.VACE]:
        if mt.get_name() in metrics_config['METRICS']:
            metrics_list.append(mt(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    return evaluator, dataset_list, metrics_list


def config_process_bdd100k(args, dataset_name, eval_config, ts_path=None):
    eval_config = deepcopy(eval_config)
    ts_path = args.output_dir if ts_path is None else ts_path
    output_folder = Path(ts_path) / args.tracker_name / 'eval'
    set_dir(output_folder)

    evaluator_config = eval_config['evaluator']
    evaluator_config['USE_PARALLEL'] = False
    evaluator_config['LOG_ON_ERROR'] = str(output_folder / 'error_log.txt')

    trackers_to_eval = [args.tracker_name, 'IPTrack']
    for tracker_name in trackers_to_eval:
        convert_motchallenge2bdd100k(eval_config['coco_path'], ts_path=ts_path, tracker_name=tracker_name, sub_folder="track_results")

    dataset_config = eval_config['dataset']

    dataset_config_new = {
        'BENCHMARK' : dataset_name,
        "GT_FOLDER": str(eval_config['gt_path'] + '/gt'),
        'TRACKERS_FOLDER' : ts_path,
        'TRACKERS_TO_EVAL': trackers_to_eval,
        'TRACKER_SUB_FOLDER': 'bdd100k_result',
        "OUTPUT_FOLDER" : str(output_folder)
    }
    dataset_config.update(dataset_config_new)
    metrics_config = eval_config['metrics']

    evaluator = trackeval.Evaluator(evaluator_config)
    dataset_class_func = get_eval_dataset_func(eval_config['dataset_class'])
    dataset_list = [dataset_class_func(dataset_config)]

    metrics_list = []
    for mt in [metrics.HOTA, metrics.CLEAR, metrics.Identity]:
        if mt.get_name() in metrics_config['METRICS']:
            metrics_list.append(mt(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    return evaluator, dataset_list, metrics_list


def convert_motchallenge2bdd100k(coco_path, ts_path, tracker_name, sub_folder):
    ts_path = Path(ts_path)
    txt_path = ts_path / tracker_name / sub_folder
    coco = MotCOCO(coco_path)

    # seq_save_path = ts_path / tracker_name / 'bdd100k_result' / 'track_result.json'
    # seq_save_path.parent.mkdir(parents=True, exist_ok=True)

    seq_save_path = ts_path / tracker_name / 'bdd100k_result' 
    set_dir(seq_save_path)

    all_results = {}

    for video_name, video_info in coco.videoFrameToImg.items():
        all_results[video_name] = {}
        for fid, img_info in video_info.items():
            img_dir = img_info['file_name']
            json_result = {}
            json_result['videoName'] = video_name
            json_result['name'] = Path(img_dir).name
            json_result["frameIndex"] = fid - 1
            json_result['labels'] = []
            all_results[video_name][fid] = json_result

    for txt_dir in txt_path.glob('*.txt'):
        seq_name = txt_dir.stem
        frame_info = all_results[seq_name]

        try:
            input_ = np.loadtxt(str(txt_dir), delimiter=',')
            for box_info in input_:
                fid = int(box_info[0])
                track_id = int(box_info[1])
                x1, y1, w, h = box_info[2], box_info[3], box_info[4], box_info[5]
                x2, y2 = x1 + w, y1 + h
                score = box_info[6]
                class_id = int(box_info[7])

                l_dict = {}
                l_dict['category'] = coco.cats[class_id]['name']
                l_dict['box2d'] = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                l_dict['id'] = track_id

                frame_info[fid]['labels'].append(l_dict)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(str(e))
    
    for seq_name, frame_info in all_results.items():
        video_results_path = seq_save_path / '{}.json'.format(seq_name)
        
        jsons = [v for _, v in frame_info.items()]
    
        with open(str(video_results_path), 'w') as f:
                json.dump(jsons, f)
        
