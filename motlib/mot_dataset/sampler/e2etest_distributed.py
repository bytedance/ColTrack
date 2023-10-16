# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from torch.utils.data import DistributedSampler
import math
from typing import TypeVar, Optional, Iterator
from motlib.mot_dataset.data_manager.mot2coco import MOT2CoCoDataset
import torch
from copy import deepcopy
import numpy as np

class MOTE2ETestDistributedSampler(DistributedSampler):
    def __init__(self, dataset: MOT2CoCoDataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        worker_idx_list, worker_valid_list, max_img_num = self.init()

        self.num_samples = max_img_num
        self.total_size = self.num_samples * self.num_replicas
        self.worker_idx_list = worker_idx_list
        self.worker_valid_list = worker_valid_list
        self.valid_num = self.worker_valid_list[self.rank]

        assert not self.drop_last
        assert not self.shuffle
    
    def __iter__(self):

        indices = self.worker_idx_list[self.rank]
        assert len(indices) == self.num_samples
        assert len(indices) >= self.valid_num

        return iter(indices)
    
    def init(self):
        # worker_video_list = [[] for _ in range(self.num_replicas)]
        worker_idx_list = [[] for _ in range(self.num_replicas)]
        worker_valid_list = [0 for _ in range(self.num_replicas)]
        video_ids = sorted(list(self.dataset.coco.videoToImgs.keys()))
        video_info = [[v, self.dataset.coco.videos[v]['number']] for v in video_ids]
        video_info = sorted(video_info, key=lambda x:x[1])
        
        for i, (vid, frame_num) in enumerate(video_info):
            worker_id = i % self.num_replicas
            # worker_video_list[worker_id].append(vid)
            worker_valid_list[worker_id] += frame_num
            img_ids = [img['id'] for img in self.dataset.coco.videoToImgs[vid]]
            worker_idx_list[worker_id].extend(img_ids)
            assert len(worker_idx_list[worker_id]) == worker_valid_list[worker_id]
        
        worker_valid_list = np.asarray(worker_valid_list)
        max_img_num = worker_valid_list.max()
        for worker_id in range(self.num_replicas):
            pad_size = max_img_num - worker_valid_list[worker_id]
            worker_idx_list[worker_id].extend(list(range(pad_size)))
            assert len(worker_idx_list[worker_id]) == max_img_num

        assert worker_valid_list.sum() == len(self.dataset.coco.imgs)

        return worker_idx_list, worker_valid_list, max_img_num





