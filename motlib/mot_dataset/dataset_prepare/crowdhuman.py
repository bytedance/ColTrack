# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.mot_dataset.dataset_prepare.dataset_base import DatasetFileBase
from motlib.mot_dataset.dataset_prepare import MOTDATASETFILE

from pathlib import Path
import numpy as np
from copy import deepcopy
import json
import os
import cv2


__all__ = ['Crowdhuman']


@MOTDATASETFILE.register()
class Crowdhuman(DatasetFileBase):
    def init(self):
        super().init()
        self.subdirectory = 'crowdhuman'
        self.data_path = self.data_root / self.subdirectory

    def load(self):
        data_path = self.data_path
        output = {}
        output['train'] = load_dataset_core(data_path / 'images', 'train', ann_dir= data_path / 'annotation_train.odgt')
        output['val']= load_dataset_core(data_path / 'images', 'val', ann_dir= data_path / 'annotation_val.odgt')
        return output


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def load_dataset_core(data_path, split_name, ann_dir):
    data_path = Path(data_path)
    anns_data = load_func(ann_dir)

    tid_curr = 0
    videos_info = []
    for i, ann_data in enumerate(anns_data):
        image_name = '{}.jpg'.format(ann_data['ID'])
        img_path = str(data_path / split_name / image_name)
        anns = ann_data['gtboxes']
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]
        
        video_info = {'height':img_height,'width': img_width, 'frame_rate':-1, 'dataset': 'Crowdhuman', 'name': ann_data['ID']}
        frame_info = {}
        frame_info[0] = {'dir': 'images/' + split_name + '/' + image_name, 'boxs':[]}
        
        for i in range(len(anns)):
            if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                continue
            x, y, w, h = anns[i]['fbox']
            x += w / 2
            y += h / 2
            frame_info[0]['boxs'].append([tid_curr, -1, x / img_width, y / img_height, w / img_width, h / img_height, 1])
     
            tid_curr += 1
        video_info['frames']=frame_info
        video_info['frame_num'] = 1
        videos_info.append(video_info)
    output = {'videos':videos_info, 'track_num':tid_curr, 'video_num': len(videos_info), 'frame_num':len(anns_data), 'id_num': -1}
    return output

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    output = Crowdhuman('').load()
    for k, v in output.items():
        print(k)

        print(v.get('track_num', -1000))
        print(v.get('video_num', -1000))
        print(v.get('frame_num', -1000))
        print(v.get('id_num', -1000))