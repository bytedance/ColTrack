# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import json
import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from copy import deepcopy
from prettytable import PrettyTable
import logging

from motlib.mot_dataset.dataset_prepare.utils.util import load_info

class MOT2CoCoDataset(Dataset):
    def __init__(self, args, train_or_test, transforms=None) -> None:
        self.args = args
        self.split_name = args.dataset_file
        self._transforms = transforms
        self.train_or_test = train_or_test
        self.coco  = load_info(args, split_name=self.split_name, train_or_test=train_or_test, data_type='coco')
        self.gt_path = self.coco.eval_config['gt_path']
        self.num_classes = self.coco.num_classes
        self.dataset_name = self.coco.dataset_name
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._infoprint()
    
    def _infoprint(self):
        video_ids = list(self.coco.videoToImgs.keys())
        table = None
        img_num = 0
        dataset_info = {}
        dataset_video_num = {}
        video_num_limit = 30

        for vid in video_ids:
            video_info_i = deepcopy(self.coco.videos[vid])
            img_num += video_info_i['number']
            if table is None:
                table_k = list(video_info_i.keys())
                table_k = ['name'] + [k for k in table_k if k != 'name']
                table = PrettyTable([str(k) for k in table_k])
            
            dataset_video_num[video_info_i['dataset']] = dataset_video_num.get(video_info_i['dataset'], 0) + 1
            
            if video_info_i['number'] > 1:
                if dataset_video_num[video_info_i['dataset']] == video_num_limit:
                    table.add_row([ '...' if k != 'dataset' else str(video_info_i[k]) for k in table_k])
                elif dataset_video_num[video_info_i['dataset']] < video_num_limit:
                    table.add_row([str(video_info_i[k]) for k in table_k])
            dataset_info[video_info_i['dataset']] = dataset_info.get(video_info_i['dataset'], 0) + video_info_i['number']
            
    
        total_info = {}
        for k in table_k:
            if k =='number':
                total_info[k] = img_num
            elif k == 'name':
                total_info[k] = 'Total'
            else:
                total_info[k] = '-'
 
        for d, v in dataset_info.items():
            d_info = {}
            for k in table_k:
                if k == 'dataset':
                    d_info[k] = d
                elif k == 'name':
                    d_info[k] = 'Dataset'
                elif k == 'number':
                    d_info[k] = v
                else:
                    d_info[k] = '-'
            table.add_row([str(d_info[k]) for k in table_k])
        table.add_row([str(total_info[k]) for k in table_k])
        logger = logging.getLogger(__name__)
        logger.info('{} video info of {} Split\n{}'.format(self.train_or_test, self.split_name, table))
            
    
    def _prepare(self, img_info, anno):
        img_dir, image_id, frame_id, video_id = img_info['file_name'], img_info['id'], img_info['frame_id'], img_info['video_id']
        raw_h, raw_w = img_info['height'], img_info['width']
        try:
            image = Image.open(img_dir).convert('RGB')
        except:
            print(img_dir)
            raise KeyError
        w, h = image.size

        assert w == raw_w and raw_h == h, 'img h {} w {}. record h {} w {}'.format(h, w, raw_h, raw_w)

        image_id = torch.tensor([image_id])
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        assert (boxes[:, 2:] >= boxes[:, :2]).all()

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        track_ids = [obj["track_id"] for obj in anno]
        track_ids = torch.tensor(track_ids, dtype=torch.int64)

        instance_ids = [obj["instance_id"] for obj in anno]
        instance_ids = torch.tensor(instance_ids, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        track_ids = track_ids[keep]
        instance_ids = instance_ids[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # for mot
        target['frame_id'] = torch.tensor([frame_id])
        target['video_id'] = torch.tensor([video_id ])
        target['track_id'] = track_ids
        target['instance_id'] = instance_ids

        return image, target

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        img, target = self._prepare(img_info, anno)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    
    def __len__(self):
        return len(self.coco.imgs)
        