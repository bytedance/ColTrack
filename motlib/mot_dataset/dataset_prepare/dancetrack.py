# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.mot_dataset.dataset_prepare.dataset_base import DatasetFileBase
from motlib.mot_dataset.dataset_prepare import MOTDATASETFILE

from pathlib import Path
import numpy as np
from copy import deepcopy
from .mot17 import get_fr_split


__all__ = ['DanceTrack']


@MOTDATASETFILE.register()
class DanceTrack(DatasetFileBase):
    def init(self):
        super().init()
        self.subdirectory = 'dancetrack'
        self.data_path = self.data_root / self.subdirectory

    def load(self):
        train_info = load_dataset_core(self.data_path, 'train')
        test_info = load_dataset_core(self.data_path, 'test')
        val_info = load_dataset_core(self.data_path, 'val')
        fr_val = get_fr_split(deepcopy(val_info))
    
        output = {'train':train_info, 'test':test_info, 'val': val_info}
        output.update(fr_val)
        return output


def load_dataset_core(data_path, split, name_key="dancetrack*"):
    data_path = Path(data_path)

    tid_curr = 0
    tid_last = -1
    
    videos_info = []
    split_path = data_path / split

    video_dirs = sorted(split_path.glob(name_key))
    frame_num = 0
    for video_dir in video_dirs:
        with open(video_dir / 'seqinfo.ini', 'r') as f:
            seq_info = f.read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        frame_rate = int(seq_info[seq_info.find('frameRate=') + 10:seq_info.find('\nseqLength')])

        video_info = {'height':seq_height,'width': seq_width,'frame_rate':frame_rate, 'dataset': 'DanceTrack', 'name':str(video_dir.name)}
        frame_info = {}
        if split in ['train', 'val']:
            gt_txt = video_dir / 'gt/gt.txt'
            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            
            for fid, tid, x, y, w, h, mark, label, _ in gt:
                fid = int(fid)
                tid = int(tid)
                if not tid == tid_last:
                    tid_curr += 1
                    tid_last = tid

                x += w / 2
                y += h / 2
                img_fpath =  split + '/' + str(video_dir.name) + '/img1/{:08d}.jpg'.format(fid)
                if not fid in frame_info:
                    frame_info[fid] = {'dir': str(img_fpath), 'boxs':[]}
                frame_info[fid]['boxs'].append([tid_curr, -1, x / seq_width, y / seq_height, w / seq_width, h / seq_height, 1])
        elif split == 'test':
            imgs_dir = (video_dir / 'img1').glob('*.jpg')
            for fid, img_dir in enumerate(sorted(imgs_dir)):
                img_fpath = str(img_dir)
                assert str(data_path) in img_fpath
                img_fpath = img_fpath[len(str(data_path))+1:]
                frame_info[fid] = {'dir': img_fpath}
        else:
            raise KeyError
        video_info['frames']=frame_info
        video_info['frame_num'] = len(frame_info)
        videos_info.append(video_info)
        frame_num += len(video_info['frames'])
    if split in ['train', 'val']:
        output = {'videos':videos_info, 'track_num':tid_curr, 'video_num': len(videos_info), 'frame_num':frame_num, 'id_num': -1}
    elif split == 'test':
        output = {'videos':videos_info, 'video_num': len(videos_info), 'frame_num':frame_num}
    else:
        raise KeyError
    return output


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    output = DanceTrack('')()
    for k, v in output.items():
        print(k)

        print(v.get('track_num', -1000))
        print(v.get('video_num', -1000))
        print(v.get('frame_num', -1000))
        print(v.get('id_num', -1000))