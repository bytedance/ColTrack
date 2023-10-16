# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.mot_dataset.dataset_prepare.dataset_base import DatasetFileBase
from motlib.mot_dataset.dataset_prepare import MOTDATASETFILE

from pathlib import Path
import numpy as np
from copy import deepcopy


__all__ = ['MOT17']


@MOTDATASETFILE.register()
class MOT17(DatasetFileBase):
    def init(self):
        super().init()
        self.subdirectory = 'MOT17/images'
        self.data_path = self.data_root / self.subdirectory
        self.name_key = "MOT17*FRCNN"

    def load(self):
        train_info = load_dataset_core(self.data_path, 'train', self.name_key)
        test_info = load_dataset_core(self.data_path, 'test', self.name_key)
        half_info = get_half_split(deepcopy(train_info))
        output = {'train':train_info, 'test':test_info}
        output.update(half_info)
        fp_info = get_fr_split(deepcopy(half_info['val_half']))
        output.update(fp_info)
        return output

def load_dataset_core(data_path, split, name_key="MOT17*FRCNN"):
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

        video_info = {'height':seq_height,'width': seq_width,'frame_rate':frame_rate, 'dataset': name_key[:5], 'name':str(video_dir.name)}
        frame_info = {}
        if split == 'train':
            gt_txt = video_dir / 'gt/gt.txt'
            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            
            for fid, tid, x, y, w, h, mark, label, _ in gt:
                if mark == 0 or not label == 1:
                    continue
                fid = int(fid)
                tid = int(tid)
                if not tid == tid_last:
                    tid_curr += 1
                    tid_last = tid

                x += w / 2
                y += h / 2
                img_fpath =  split + '/' + str(video_dir.name) + '/img1/{:06d}.jpg'.format(fid)
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
    if split == 'train':
        output = {'videos':videos_info, 'track_num':tid_curr, 'video_num': len(videos_info), 'frame_num':frame_num, 'id_num': -1}
    elif split == 'test':
        output = {'videos':videos_info, 'video_num': len(videos_info), 'frame_num':frame_num}
    else:
        raise KeyError
    return output


def load_dataset(data_path='', name_key="MOT17*FRCNN"):
    train_info = load_dataset_core(data_path, 'train', name_key)
    test_info = load_dataset_core(data_path, 'test', name_key)
    half_info = get_half_split(train_info)
    output = {'train':train_info, 'test':test_info}
    output.update(half_info)
    fp_info = get_fr_split(deepcopy(half_info['val_half']))
    output.update(fp_info)
    return output


def get_half_split(train_info):
    videos_info = train_info['videos']
    splits = ['train_half', 'val_half']
    output = {}
    for split in splits:
        split_videos_info = []
        frame_num = 0
        tid_set = set()
        for video_info in videos_info:
            split_video_info = {'height':video_info['height'],'width': video_info['width'],'frame_rate':video_info['frame_rate'], 'dataset': video_info['dataset'], 'name':video_info['name']}
            frame_info = video_info['frames']
            split_frame_info = {}
            num_images = len(frame_info)

            image_range = [0, num_images // 2] if 'train' in split else [num_images // 2 + 1, num_images - 1]

            for i, fid in enumerate(sorted(frame_info.keys())):
                if i < image_range[0] or i > image_range[1]:
                    continue
                split_frame_info[fid] = deepcopy(frame_info[fid])
                for box in split_frame_info[fid]['boxs']:
                    tid_set.add(box[0])
                frame_num += 1
            split_video_info['frames'] = split_frame_info
            split_video_info['frame_num'] = len( split_frame_info)
            split_videos_info.append(split_video_info)

        output[split] = {'videos':split_videos_info, 'track_num':len(tid_set), 'video_num': len(split_videos_info), 'frame_num':frame_num, 'id_num':len(tid_set)}
    return output


def get_fr_split(val_half):
    videos_info = val_half['videos']
    splits_fr = [2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 25, 30, 36, 50, 60, 90]
    output = {}
    for split_fr in splits_fr:
        split_name = 'val_{}'.format(split_fr)
        split_videos_info = []
        frame_num = 0
        tid_set = set()
        for video_info in videos_info:
            split_video_info = {'height':video_info['height'],'width': video_info['width'],'frame_rate':video_info['frame_rate'] // split_fr, 'dataset': video_info['dataset'], 'name':video_info['name']}
            frame_info = video_info['frames']
            split_frame_info = {}
            num_images = len(frame_info)

            fids = sorted(frame_info.keys())

            for i in range(0, num_images, split_fr):
                fid = fids[i]
                split_frame_info[fid] = deepcopy(frame_info[fid])
                if 'boxs' in split_frame_info[fid]:
                    for box in split_frame_info[fid]['boxs']:
                        tid_set.add(box[0])
                frame_num += 1
            split_video_info['frames'] = split_frame_info
            split_video_info['frame_num'] = len(split_frame_info)
            split_videos_info.append(split_video_info)

        output[split_name] = {'videos':split_videos_info, 'track_num':len(tid_set), 'video_num': len(split_videos_info), 'frame_num':frame_num, 'id_num':len(tid_set)}
    return output



if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    output = MOT17('').split_dict
    for k, v in output.items():
        print(k)

        print(v.get('track_num', -1000))
        print(v.get('video_num', -1000))
        print(v.get('frame_num', -1000))
        print(v.get('id_num', -1000))