# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.mot_dataset.dataset_prepare.dataset_base import DatasetFileBase
from motlib.mot_dataset.dataset_prepare import MOTDATASETFILE

from pathlib import Path
import numpy as np
from copy import deepcopy
import json


__all__ = ['PersonPath22']


@MOTDATASETFILE.register()
class PersonPath22(DatasetFileBase):
    def init(self):
        super().init()
        self.subdirectory = 'personpath22'
        self.data_path = self.data_root / self.subdirectory

    def load(self):

        json_path = self.data_path / 'annotation/anno_visible_2022.json'
        dataset = MotionDataset(self.data_path, json_path)

        output = dataset.prepare_data()
        self.category = dataset.category
        return output


class MotionDataset(object):
    def __init__(self, data_path, json_path) -> None:
        self.data_path = Path(data_path)
        self.json_path = Path(json_path)
        self.anns_path = self.json_path.parent / self.json_path.stem
        self.split_path = self.data_path / 'annotation' / 'splits.json'
        self.prepare_category()
    
    def prepare_category(self):
        category = [
                {'supercategory': 'none', 'id': 1, 'name': 'pedestrian'}, 
                {'supercategory': 'none', 'id': 2, 'name': 'person_on_vehicle'}, 
                {'supercategory': 'none', 'id': 3, 'name': 'car'}, 
                {'supercategory': 'none', 'id': 4, 'name': 'bicycle'}, 
                {'supercategory': 'none', 'id': 5, 'name': 'motorbike'}, 
                {'supercategory': 'none', 'id': 6, 'name': 'non_mot_vehicle'}, 
                {'supercategory': 'none', 'id': 7, 'name': 'static_person'}, 
                {'supercategory': 'none', 'id': 8, 'name': 'distractor'}, 
                {'supercategory': 'none', 'id': 9, 'name': 'occluder'}, 
                {'supercategory': 'none', 'id': 10, 'name': 'occluder_on_ground'}, 
                {'supercategory': 'none', 'id': 11, 'name': 'occluder_full'}, 
                {'supercategory': 'none', 'id': 12, 'name': 'reflection'}, 
                {'supercategory': 'none', 'id': 13, 'name': 'crowd'}
                ]
        
        category2id_map = {}
        id2category_map = {}
        for cls in category:
            category2id_map[cls['name']] = cls['id']
            id2category_map[int(cls['id'])] = cls['name']
        
        self.category2id_map = category2id_map
        self.id2category_map = id2category_map
        self.category = category
    
    def prepare_data(self):
        video_info = self.get_video_info()
        split_data = MotionDataset.load_data(self.split_path)
        res = {}
        for k, v in split_data.items():
            split_video = [video_info[video_name] for video_name in v]
            frame_num = [video_info['frame_num'] for video_info in split_video]
            frame_num = sum(frame_num)
            res[k] = {'videos': split_video, 'video_num': len(split_video), 'frame_num':frame_num}
        return res
    
    def get_video_info(self):
        res = {}
        videos_info = MotionDataset.load_data(self.json_path)['samples']
        for k, v in videos_info.items():
            video_info = v['metadata']
            video_name = str(k).split('.')[0]
            seq_width = video_info['resolution']['width']
            seq_height = video_info['resolution']['height']
            res_video_info = {'height': seq_height,'width': seq_width,'frame_rate':video_info['fps'], 'dataset': 'PersonPath22', 'name': video_name}
            anns_dir = self.anns_path / (video_info['data_path'] + '.json')
            anns_data = MotionDataset.load_data(anns_dir)
            boxes_info = anns_data['entities']
            frame_info = {}
            for box_info in boxes_info:
                # {'person': 1, 'severly_occluded_person':2 , 'person_in_background':3, 'sitting_person':4, 'person_on_open_vehicle':5, 'standing_person':6, 'person_in_vehicle':7}
                if 'pedestrian' not in box_info['labels']:
                    continue
                fid = box_info['blob']['frame_idx'] + 1
                tid = box_info['id']
                box = box_info['bb']
                img_fpath = 'frame_data/{}/{:0>6d}.jpg'.format(video_name,fid)
                if fid not in frame_info:
                    frame_info[fid] = {'dir': str(img_fpath), 'boxs':[]}
                x, y, w, h = box
                x += w / 2
                y += h / 2
                frame_info[fid]['boxs'].append([tid, -1, x / seq_width, y / seq_height, w / seq_width, h / seq_height, self.category2id_map[box_info['labels']]])
            for fid, finfo in frame_info.items():
                if len(finfo['boxs']) == 0:
                    frame_info.pop(fid)
            res_video_info['frames']=frame_info
            res_video_info['frame_num'] = len(frame_info)
            res[k] = res_video_info
        return res

    @staticmethod
    def load_data(data_dir):
        with open(str(data_dir), 'r') as f:
            data = json.load(f)
        return data