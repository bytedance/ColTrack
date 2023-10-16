#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

import cv2
import numpy as np

from motlib.mot_dataset.transform.yolox.utils import adjust_box_anns

import random
from copy import deepcopy

from motlib.mot_dataset.transform.yolox.data_augment import box_candidates, augment_hsv
from motlib.mot_dataset.transform.yolox.dataset import Dataset
from motlib.mot_dataset.transform.yolox.mosaic import MosaicDetection, get_mosaic_coordinate
from collections import defaultdict

from .data_augment import random_perspective, RandomErasing


class MOTMosaicDetection(MosaicDetection):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, 
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True, args=None, train_or_test="train", transforms=None
    ):
        super().__init__(dataset, img_size, mosaic, degrees, translate, scale, mscale, shear, perspective, enable_mixup, args, train_or_test, transforms)
        self.erasing_func = RandomErasing(p=args.p_era, area_keep=args.area_keep, value=114)

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic:
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            imgs_all = defaultdict(list)
            labels_all = defaultdict(list)
            img_info_all = defaultdict(list)
            img_id_all = defaultdict(list)

            mosaic_img_all = []
            mosaic_label_all = []

            for i_mosaic, index in enumerate(indices):
                imgs, _labels_all, img_info_i, img_id_i = self._dataset.pull_video(index)
                for frame_i, (img, label, img_info, img_id) in enumerate(zip(imgs, _labels_all, img_info_i, img_id_i)):
                    imgs_all[frame_i].append(img)
                    labels_all[frame_i].append(label)
                    img_info_all[frame_i].append(img_info)
                    img_id_all[frame_i].append(img_id)
                
            for frame_i in sorted(list(imgs_all.keys())):
                mosaic_labels = []
                for i_mosaic, index in enumerate(indices):
                    img, _labels, = imgs_all[frame_i][i_mosaic], labels_all[frame_i][i_mosaic]

                    h0, w0 = img.shape[:2]  # orig hw
                    scale = min(1. * input_h / h0, 1. * input_w / w0)
                    img = cv2.resize(
                        img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                    )
                    # generate output mosaic image
                    (h, w, c) = img.shape[:3]
                    if i_mosaic == 0:
                        mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                    # suffix l means large image, while s means small image in mosaic aug.
                    (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                        mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                    )

                    mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                    padw, padh = l_x1 - s_x1, l_y1 - s_y1

                    labels = _labels.copy()
                    # Normalized xywh to pixel xyxy format
                    if _labels.size > 0:
                        labels[:, 0] = scale * _labels[:, 0] + padw
                        labels[:, 1] = scale * _labels[:, 1] + padh
                        labels[:, 2] = scale * _labels[:, 2] + padw
                        labels[:, 3] = scale * _labels[:, 3] + padh
                    mosaic_labels.append(labels)

                if len(mosaic_labels):
                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    '''
                    np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                    np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                    np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                    np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
                    '''
                    
                    mosaic_labels = mosaic_labels[mosaic_labels[:, 0] < 2 * input_w]
                    mosaic_labels = mosaic_labels[mosaic_labels[:, 2] > 0]
                    mosaic_labels = mosaic_labels[mosaic_labels[:, 1] < 2 * input_h]
                    mosaic_labels = mosaic_labels[mosaic_labels[:, 3] > 0]
                
                mosaic_img_all.append(mosaic_img)
                mosaic_label_all.append(mosaic_labels)
            
            #augment_hsv(mosaic_img)
            box_num = []
            for mosaic_label_i in mosaic_label_all:
                box_num.append(len(mosaic_label_i))

            for _ in range(100):
                mosaic_img_all_tmp, mosaic_label_all_tmp = random_perspective(
                    mosaic_img_all,
                    mosaic_label_all,
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear,
                    perspective=self.perspective,
                    border=[-input_h // 2, -input_w // 2],
                )  # border to remove

                zero_label_flag = False
                for i_frame, mosaic_label_i in enumerate(mosaic_label_all_tmp):
                    if mosaic_label_i.shape[0] == 0 and box_num[i_frame] > 0:
                        zero_label_flag = True
                        break
                if not zero_label_flag:
                    mosaic_img_all, mosaic_label_all = mosaic_img_all_tmp, mosaic_label_all_tmp
                    break
            
            zero_label_flag = False
            for mosaic_label_i in mosaic_label_all:
                if len(mosaic_label_i) == 0:
                    zero_label_flag = True
                    break
            
            if self.enable_mixup and not zero_label_flag:
                mosaic_img_all, mosaic_label_all, img_info_mixup, img_id_mixup = self.mixup(mosaic_img_all, mosaic_label_all, self.input_dim)
                for frame_i, (img_info_i, img_id_i) in enumerate(zip(img_info_mixup, img_id_mixup)):
                    img_info_all[frame_i].append(img_info_i)
                    img_id_all[frame_i].append(img_id_i)
            
            mosaic_img_all, mosaic_label_all = self.erasing_func(mosaic_img_all, mosaic_label_all)

            mosaic_img_all, mosaic_label_all = self._transforms(mosaic_img_all, mosaic_label_all, self.input_dim)


            return mosaic_img_all, mosaic_label_all, img_info_all, img_id_all

        else:
            self._dataset._input_dim = self.input_dim
            imgs, _labels_all, img_info_i, img_id_i = self._dataset.pull_video(idx)
            
            imgs, _labels_all = self._transforms(imgs, _labels_all, self.input_dim)
            return imgs, _labels_all, img_info_i, img_id_i

    def mixup(self, origin_img_all, origin_labels_all, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        x_offset, y_offset = None, None

        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img_all, cp_labels_all, img_info, img_id = self._dataset.pull_video(cp_index)

        outout_origin_img = []
        output_origin_labels = []

        for i_frame, (origin_img, origin_labels) in enumerate(zip(origin_img_all, origin_labels_all)):
            assert len(origin_labels) != 0
            img = img_all[i_frame]
            cp_labels = cp_labels_all[i_frame]

            if len(img.shape) == 3:
                cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
            else:
                cp_img = np.ones(input_dim) * 114.0
            cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            cp_img[
                : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
            ] = resized_img
            cp_img = cv2.resize(
                cp_img,
                (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
            )
            cp_scale_ratio *= jit_factor
            if FLIP:
                cp_img = cp_img[:, ::-1, :]

            origin_h, origin_w = cp_img.shape[:2]
            target_h, target_w = origin_img.shape[:2]
            padded_img = np.zeros(
                (max(origin_h, target_h), max(origin_w, target_w), 3)
            ).astype(np.uint8)
            padded_img[:origin_h, :origin_w] = cp_img

            if x_offset is None:
                x_offset, y_offset = 0, 0
                if padded_img.shape[0] > target_h:
                    y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
                if padded_img.shape[1] > target_w:
                    x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
            padded_cropped_img = padded_img[
                y_offset: y_offset + target_h, x_offset: x_offset + target_w
            ]

            cp_bboxes_origin_np = adjust_box_anns(
                cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
            )
            if FLIP:
                cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
                )
            cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
            '''
            cp_bboxes_transformed_np[:, 0::2] = np.clip(
                cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
            )
            cp_bboxes_transformed_np[:, 1::2] = np.clip(
                cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
            )
            '''
            cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
            cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
            keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

            if keep_list.sum() >= 1.0:
                cls_labels = cp_labels[keep_list, 4:5].copy()
                id_labels = cp_labels[keep_list, 5:6].copy()
                box_labels = cp_bboxes_transformed_np[keep_list]
                labels = np.hstack((box_labels, cls_labels, id_labels))
                # remove outside bbox
                labels = labels[labels[:, 0] < target_w]
                labels = labels[labels[:, 2] > 0]
                labels = labels[labels[:, 1] < target_h]
                labels = labels[labels[:, 3] > 0]
                origin_labels = np.vstack((origin_labels, labels))
                origin_img = origin_img.astype(np.float32)
                origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
            
            outout_origin_img.append(origin_img)
            output_origin_labels.append(origin_labels)

        return outout_origin_img, output_origin_labels, img_info, img_id
