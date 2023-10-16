# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from copy import deepcopy
import logging
import numpy as np
from functools import wraps
from motlib.mot_dataset.transform.yolox.dataset import YOLODataset, Dataset
from .data_augment import FixedMotRandomShift


class MOTYOLODataset(YOLODataset):
    def __init__(self, args, train_or_test, transforms=None) -> None:
        super().__init__(args, train_or_test, transforms)
        
        self.sample_mode = args.sample_mode
        self.sampler_steps: list = args.sampler_steps
        if not isinstance(args.sampler_lengths, (list, tuple)):
            assert isinstance(args.sampler_lengths, int)
            args.sampler_lengths = [args.sampler_lengths for _ in range(len(self.sampler_steps))]

        self.num_frames_per_batch = max(args.sampler_lengths)

        self.sample_interval_base = args.sample_interval
        self.sampler_interval_scale = args.sampler_interval_scale
        self.sample_interval_step = [int(self.sample_interval_base * s) for s in self.sampler_interval_scale]
        self.sample_interval = max(self.sample_interval_step)

        self.item_num = len(self.coco.imgs) - (self.num_frames_per_batch - 1) * self.sample_interval
        
        self.lengths: list = args.sampler_lengths
        self.logger = logging.getLogger(__name__)
        self.logger.info("sampler_steps={} lenghts={} insterval={} interval scale={}".format(self.sampler_steps, self.lengths, self.sample_interval_step, self.sampler_interval_scale))
        
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps)
            assert len(self.sampler_interval_scale) == len(self.sampler_steps)
            assert self.sampler_steps[0] == 0
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]

            self.item_num = len(self.coco.imgs) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0
            self.padding_scale = self.sampler_interval_scale[0]
        
        self.fixedMotRandomShift = FixedMotRandomShift()
    
    def __len__(self):
        return self.item_num
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i 
        self.num_frames_per_batch = self.lengths[self.period_idx]
        self.sample_interval = self.sample_interval_step[self.period_idx]
        self.padding_scale = self.sampler_interval_scale[self.period_idx]
        self.logger.info("set epoch: epoch {} period_idx={} frames {} interval_max {} scale {}".format(epoch, self.period_idx, self.num_frames_per_batch, self.sample_interval, self.padding_scale))

    def step_epoch(self):
        # one epoch finishes.
        self.logger.info("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)
    
    def _get_sample_range(self, start_idx):
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)

        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range
    
    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        img_info = []
        img_id = []
        for i in range(start, end, interval):
            img_i, targets_i, img_info_i, img_id_i = self.pull_item(i)
            images.append(img_i)
            targets.append(targets_i)
            img_info.append(img_info_i)
            img_id.append(img_id_i)
        return images, targets, img_info, img_id
    
    def dataset_specify_process(self, images, targets, img_info, img_id):
        shift_flag = False
        h, w = img_info[0][0], img_info[0][1]
        for i_img in range(len(images)):
            h0, w0 = img_info[i_img][0], img_info[i_img][1]
            video_id = img_info[i_img][3]
            imgs = self.coco.videoToImgs[video_id]
            if len(imgs) == 1 or h0 != h or w0 != w:
                shift_flag = True
                break
        if shift_flag:
            images, targets = self.fixedMotRandomShift(images, targets, self.padding_scale)
            img_id = [deepcopy(img_id[0]) for _ in range(len(img_id))]
            img_info = [deepcopy(img_info[0]) for _ in range(len(img_info))]
        return images, targets, img_info, img_id
    
    def pull_video(self, index):
        sample_start, sample_end, sample_interval = self._get_sample_range(index)
        images, targets, img_info, img_id = self.pre_continuous_frames(sample_start, sample_end, sample_interval)
        images, targets, img_info, img_id = self.dataset_specify_process(images, targets, img_info, img_id)
        return images, targets, img_info, img_id 
    
    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        images, targets, img_info, img_id = self.pull_video(index)
        

        if self._transforms is not None:
            images, targets = self._transforms(images, targets, self.input_dim)
        return images, targets, img_info, img_id
    
    