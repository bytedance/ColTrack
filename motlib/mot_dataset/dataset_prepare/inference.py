# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.mot_dataset.dataset_prepare.dataset_base import DatasetFileBase
from motlib.mot_dataset.dataset_prepare import MOTDATASETFILE

from pathlib import Path
import numpy as np
from copy import deepcopy
import cv2
from .mot17 import get_fr_split


__all__ = ['Inference']


@MOTDATASETFILE.register()
class Inference(DatasetFileBase):
    def init(self):
        super().init()
        self.subdirectory = ''
        self.data_path = self.data_root / self.subdirectory

    def load(self):
        output = {'val_1': load_dataset_core(self.data_path, '*')}
        fr_val = get_fr_split(deepcopy(output['val_1']))
        output.update(fr_val)
        return output


def load_dataset_core(data_path, name_key='*'):
    data_path = Path(data_path)
    frame_num = 0
    videos_info = []
    for video_dir in sorted(data_path.glob(name_key)):
        imgs_dir = video_dir.glob('*.*')
        imgs_dir = sorted([d for d in imgs_dir if d.suffix in ['.jpg', '.jpeg', '.png', '.tif']])
        assert len(imgs_dir) > 0

        img = cv2.imread(
                str(imgs_dir[0]),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        img_height, img_width = img.shape[0:2]

        
        video_info = {'height':img_height,'width': img_width, 'frame_rate':-1, 'dataset': 'Inference', 'name': video_dir.name}

        frame_info = {}
        for i, img_path in enumerate(imgs_dir):
            frame_info[i] = {'dir': str(img_path)[len(str(data_path))+1:]}
            
        video_info['frames']=frame_info
        video_info['frame_num'] = len(imgs_dir)
        frame_num += len(imgs_dir)
        videos_info.append(video_info)
    output = {'videos':videos_info, 'video_num': len(videos_info), 'frame_num':frame_num}
    return output