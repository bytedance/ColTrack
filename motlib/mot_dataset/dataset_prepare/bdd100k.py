# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.mot_dataset.dataset_prepare.dataset_base import DatasetFileBase
from motlib.mot_dataset.dataset_prepare import MOTDATASETFILE

from pathlib import Path
import numpy as np
from copy import deepcopy
import json
from .mot17 import get_fr_split

from motlib.evaluation.evaluate_metrics.trackeval_manager import trackeval
from motlib.evaluation.evaluate_metrics.trackeval_manager.dataset import get_eval_dataset_func


__all__ = ['BDD100K']


@MOTDATASETFILE.register()
class BDD100K(DatasetFileBase):
    def init(self):
        self.subdirectory = 'bdd100k'
        self.data_path = self.data_root / self.subdirectory

        self.ann_path = self.data_path / 'labels/box_track_20'

        self.images_path = self.data_path / 'images/track'

        self.classes_to_eval = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle']

        self.class_name_to_class_id = {'pedestrian': 1, 'rider': 2, 'other person': 3, 'car': 4, 'bus': 5, 'truck': 6,
                                       'train': 7, 'trailer': 8, 'other vehicle': 9, 'motorcycle': 10, 'bicycle': 11}
        
        self.category = [
            {'supercategory': 'none', 'id': 1, 'name': 'pedestrian'},
            {'supercategory': 'none', 'id': 2, 'name': 'rider'},
            {'supercategory': 'none', 'id': 4, 'name': 'car'},
            {'supercategory': 'none', 'id': 5, 'name': 'bus'},
            {'supercategory': 'none', 'id': 6, 'name': 'truck'},
            {'supercategory': 'none', 'id': 7, 'name': 'train'},
            {'supercategory': 'none', 'id': 10, 'name': 'motorcycle'},
            {'supercategory': 'none', 'id': 11, 'name': 'bicycle'},
            ]


    def load(self):
        train_info = load_dataset_train_val(self.data_path, self.images_path, self.ann_path, 'train', self.class_name_to_class_id, self.classes_to_eval)
        val_info = load_dataset_train_val(self.data_path, self.images_path, self.ann_path, 'val', self.class_name_to_class_id, self.classes_to_eval)
        test_info = load_dataset_test(self.data_path, self.images_path)
        fr_val = get_fr_split(deepcopy(val_info))
        
        output = {'train':train_info, 'test':test_info, 'val': val_info}
        output.update(fr_val)
        return output
    
    def _get_eval_setting(self):
        eval_dataset_class = 'BDD100K'
        eval_dataset_func = get_eval_dataset_func(eval_dataset_class)
        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config['PRINT_ONLY_COMBINED'] = True
        default_dataset_config = eval_dataset_func.get_default_dataset_config()
        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
        cut_num = len(str(self.data_root)) + 1

        config = {
            'name': 'BDD100K',
            'evaluator': default_eval_config,
            'dataset': default_dataset_config,
            'metrics': default_metrics_config,
            'dataset_class': eval_dataset_class,
            'write_gt_func': "write_gt_bdd100k",
            'anns_path' : str((self.ann_path / 'val').resolve())[cut_num:]
            }

        return config


def load_dataset_test(data_path, images_path):
    cut_num = len(str(data_path)) + 1
    images_path = Path(images_path) / 'test'

    videos_info = []
    W = 1280
    H = 720
    frame_rate = -1
    frame_num = 0

    for video_dir in sorted(images_path.glob('*')):
        video_name = video_dir.stem
        video_info = {'height':H,'width': W,'frame_rate':frame_rate, 'dataset': 'BDD100K', 'name':video_name}
        frame_info = {}
        for fid, frame_dir in enumerate(sorted(video_dir.glob('*.jpg'))):
            img_dir = str(frame_dir.resolve())[cut_num:]
            frame_info[fid+1] = {'dir': img_dir}
        video_info['frames']=frame_info
        video_info['frame_num'] = len(frame_info)
        videos_info.append(video_info)
        frame_num += len(video_info['frames'])  

    output = {'videos':videos_info, 'video_num': len(videos_info), 'frame_num':frame_num,}
    return output




def load_dataset_train_val(data_path, images_path, ann_path, split, class_name_to_class_id, classes_to_eval):
    assert split in ['train', 'val']
    cut_num = len(str(data_path)) + 1
    images_path = Path(images_path) / split
    anns_path = Path(ann_path) / split

    W = 1280
    H = 720
    frame_rate = -1
    
    video_set = set()
    videos_info = []
    tid_set = set()
    frame_num = 0

    for anns_dir in sorted(anns_path.glob('*.json')):
        json_video_name = anns_dir.stem
        assert json_video_name not in video_set
        video_set.add(json_video_name)

        video_info = {'height':H,'width': W,'frame_rate':frame_rate, 'dataset': 'BDD100K', 'name':json_video_name}
        frame_info = {}

        with open(str(anns_dir), 'r') as f:
            labels_json = json.load(f)
        for label_json in labels_json:
            img_name = label_json['name']
            video_name = label_json['videoName']
            frame_index = label_json['frameIndex']
            fid = frame_index + 1
            assert video_name == json_video_name
            img_dir = images_path / video_name / img_name
            img_dir = str(img_dir.resolve())[cut_num:]
            assert fid not in frame_info
            frame_info[fid] = {'dir': img_dir, 'boxs':[]}

            labels = label_json['labels']
            for label in labels:
                category = label['category']
                if category not in classes_to_eval:
                    continue
                x1 = label['box2d']['x1']
                x2 = label['box2d']['x2']
                y1 = label['box2d']['y1']
                y2 = label['box2d']['y2']
                width = x2 - x1
                height = y2 - y1
                x_center = (x1+x2)/2./ W
                y_center = (y1+y2)/2./ H
                width /= W
                height /= H 
                identity = int(label['id'])
                tid_set.add(identity)
                frame_info[fid]['boxs'].append([identity, -1, x_center, y_center, width, height, class_name_to_class_id[category]])

        video_info['frames']=frame_info
        video_info['frame_num'] = len(frame_info)
        videos_info.append(video_info)
        frame_num += len(video_info['frames'])  

    output = {'videos':videos_info, 'track_num':len(tid_set), 'video_num': len(videos_info), 'frame_num':frame_num, 'id_num': -1}
    return output