# Copyright (2023) Bytedance Ltd. and/or its affiliates 



from motlib.mot_dataset.dataset_prepare.dataset_base import DatasetFileBase
from motlib.mot_dataset.dataset_prepare import MOTDATASETFILE

from pathlib import Path
import numpy as np
from copy import deepcopy
import json
import os
import cv2
from PIL import Image


__all__ = ['Cityperson']


@MOTDATASETFILE.register()
class Cityperson(DatasetFileBase):
    def init(self):
        super().init()
        self.subdirectory = 'Cityscapes'
        self.data_path = self.data_root / self.subdirectory


    def load(self):
        output = load_dataset_core(self.data_path)
        return output


def load_paths(data_path):
    with open(data_path, 'r') as file:
        img_files = file.readlines()
        img_files = [x.replace('\n', '').replace('Cityscapes/', '') for x in img_files]
        img_files = list(filter(lambda x: len(x) > 0, img_files))
    label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt') for x in img_files]
    return img_files, label_files  


def load_dataset_core(data_path):
    cur_dir = Path(os.path.realpath(__file__)).parent
    img_paths, label_paths  = load_paths(str(cur_dir / 'data_path/citypersons.train'))
    
    videos_info = []
    frame_num = 0
    tid_curr = 0
    for img_path, label_path in zip(img_paths, label_paths):
        frame_num += 1
        im = Image.open(os.path.join(data_path, img_path))
        img_height, img_width = im.size[1], im.size[0]
        assert img_height < img_width
        video_info = {'height':img_height,'width': img_width, 'frame_rate':-1, 'dataset': 'Cityperson', 'name': img_path.split('/')[-2]}
        frame_info = {}
        frame_info[0] = {'dir': img_path, 'boxs':[]}

        label_dir = os.path.join(data_path, label_path)
        if os.path.isfile(label_dir):
            labels = np.loadtxt(label_dir, dtype=np.float32).reshape(-1, 6)
            for x, y, w, h in labels[:, 2:6].tolist():
                frame_info[0]['boxs'].append([tid_curr, -1, x, y, w, h, 1])
                tid_curr += 1
        video_info['frames']=frame_info
        video_info['frame_num'] = 1
        videos_info.append(video_info)
    output = {'videos':videos_info, 'track_num':tid_curr, 'video_num': len(videos_info), 'frame_num':frame_num, 'id_num': -1}
    return {'train': output}

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    output = Cityperson('').load()
    for k, v in output.items():
        print(k)

        print(v.get('track_num', -1000))
        print(v.get('video_num', -1000))
        print(v.get('frame_num', -1000))
        print(v.get('id_num', -1000))