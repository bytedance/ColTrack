#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

import cv2
import numpy as np
import numbers
import copy
import torch
import logging
from copy import deepcopy

from motlib.mot_dataset.transform.yolox.utils import xyxy2cxcywh
from motlib.mot_dataset.transform.yolox.data_augment import TrainTransform, _distort, preproc, box_candidates

import math
import random


def random_shift(image, target, region, sizes):
    oh, ow = sizes
    i, j, h, w = region
    # step 1, shift crop and re-scale image firstly

    cropped_image = image[i:i+h,j:j+w]
    cropped_image = cv2.resize(
                        cropped_image, (ow, oh), interpolation=cv2.INTER_LINEAR
                    )

    target = target.copy()

    target[:,:4] = target[:, :4] - np.asarray([j, i, j, i], dtype=target.dtype).reshape(1, 4)
    target[:,:4] *= np.asarray([[ow / w, oh / h, ow / w, oh / h]], dtype=target.dtype).reshape(1, 4)
    cropped_boxes = target[:,:4].copy().reshape(-1, 2, 2)
    max_size = np.asarray([ow, oh], dtype=target.dtype).reshape(1, 2)
    cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clip(0)
    keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
    # target[:, :4] = cropped_boxes.reshape(-1, 4)
    target = target[keep]

    return cropped_image, target


class FixedMotRandomShift(object):
    def __init__(self, bs=1, padding=50):
        self.bs = bs
        self.raw_padding = padding
        self.area_ratio = 1 / 8
        self.target_ratio = 1 / 6
    
    def get_shift(self, num_frames, w, h, padding):
        xshift = (padding * torch.rand(self.bs)).int() + 1
        xshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        yshift = (padding * torch.rand(self.bs)).int() + 1
        yshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        xshift = xshift.numpy()
        yshift = yshift.numpy()
        xshift_max = xshift * (num_frames -1)
        yshift_max = yshift * (num_frames -1)

        ymin = max(0, yshift_max[0])
        ymax = min(h, h + yshift_max[0])
        xmin = max(0, xshift_max[0])
        xmax = min(w, w + xshift_max[0])
        new_h = min((xmax-xmin) * h / w, ymax-ymin)
        new_w = min((ymax-ymin) * w / h, xmax-xmin)
        ymax = ymin + new_h
        xmax = xmin + new_w
        return xshift, yshift, xmin, xmax, ymin, ymax
    
    def add_jitter(self, xmin, xmax, ymin, ymax):
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w / 2
        cy = ymin + h / 2

        w_add = w * 16 / 1920 * torch.rand(1) * ((torch.randn(1) > 0.0).int() * 2 - 1)
        h = h / w *(w+w_add)
        w += w_add
        xmin = int(cx - w / 2)
        xmax = int(cx + w / 2)
        ymin = int(cy - h / 2)
        ymax = int(cy + h / 2)
        return xmin, xmax, ymin, ymax

    def crop_imgs(self, imgs, targets, padding, n_frames):
        h, w = imgs.shape[:2]
        xshift, yshift, xmin_last, xmax_last, ymin_last, ymax_last = self.get_shift(n_frames, w, h, padding)
        area_crop = (xmax_last - xmin_last) * (ymax_last - ymin_last)
        if area_crop / (h*w) < self.area_ratio or (ymax_last - ymin_last) <= 10 or (xmax_last - xmin_last) <= 10:
            return None, None
        raw_img = imgs
        raw_target = targets

        res_img = []
        res_target = []

        for idx in range(n_frames):
            region = (int(ymin_last), int(xmin_last), int(ymax_last - ymin_last), int(xmax_last - xmin_last))
            img_i, target_i = random_shift(raw_img.copy(), raw_target.copy(), region, (h, w))
            if target_i.shape[0] < 1:
                return None, None
            res_img.append(img_i)
            res_target.append(target_i)

            xmin_last, xmax_last, ymin_last, ymax_last = self.add_jitter(xmin_last, xmax_last, ymin_last, ymax_last)
            
            ymin_last = max(0, ymin_last-yshift[0])
            ymax_last = min(h, ymax_last - yshift[0])
            xmin_last = max(0, xmin_last-xshift[0])
            xmax_last = min(w, xmax_last - xshift[0])
        return res_img, res_target

    def shift_imgs(self, shift_func, imgs, targets, padding_scale, n_frames):
        padding = max(int(padding_scale * self.raw_padding), 5)
        raw_target_num = targets.shape[0]
        for _ in range(100):    
            res_img, res_target = shift_func(imgs, targets, padding, n_frames=n_frames)
            if res_img is not None:
                shift_succeed = True
                for target_i in res_target:
                    if target_i.shape[0] / raw_target_num < self.target_ratio:
                        shift_succeed = False
                        break
                if shift_succeed:
                    return res_img, res_target

            padding_scale = max(0.1, padding_scale - 0.05)
            padding = max(int(padding_scale * self.raw_padding), 5)
        return None, None

    def __call__(self, imgs, targets, padding_scale):
        n_frames = len(imgs)
        assert len(imgs) == len(targets)
        imgs = imgs[0]
        targets = targets[0]

        res_img, res_target = self.shift_imgs(self.crop_imgs, imgs, targets, padding_scale, n_frames)

        if res_img is not None:
            return res_img, res_target

        res_img, res_target = self.shift_imgs(self.zoom_shift, imgs, targets, padding_scale, n_frames)
        if res_img is not None:
            return res_img, res_target
        
        print('Can not shift this images')
        raise ValueError

    def zoom_shift(self, imgs: list, targets: list, padding, n_frames):
        ret_imgs = []
        ret_targets = []

        h, w = imgs.shape[:2]

        xshift = (padding * torch.rand(self.bs)).int() + 1
        xshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        yshift = (padding * torch.rand(self.bs)).int() + 1
        yshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        xshift = xshift.numpy()
        yshift = yshift.numpy()
        
        ret_imgs.append(imgs)
        ret_targets.append(targets)
        for i in range(1, n_frames):

            ymin = max(0, -yshift[0])
            ymax = min(h, h - yshift[0])
            xmin = max(0, -xshift[0])
            xmax = min(w, w - xshift[0])
            prev_img = ret_imgs[i-1].copy()
            prev_target = copy.deepcopy(ret_targets[i-1])
            region = (int(ymin), int(xmin), int(ymax - ymin), int(xmax - xmin))
            img_i, target_i = random_shift(prev_img, prev_target, region, (h, w))
            if target_i.shape[0] < 1:
                return None, None
            ret_imgs.append(img_i)
            ret_targets.append(target_i)

        return ret_imgs, ret_targets



def random_perspective(
    imgs,
    targets_all=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    img = imgs[0]
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        new_imgs = []
        for img in imgs:
            if perspective:
                img = cv2.warpPerspective(
                    img, M, dsize=(width, height), borderValue=(114, 114, 114)
                )
            else:  # affine
                img = cv2.warpAffine(
                    img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
                )
            new_imgs.append(img)
    imgs = new_imgs

    # Transform label coordinates
    new_targets_all = []
    for targets in targets_all:
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip boxes
            #xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            #xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, :4] = xy[i]
            
            targets = targets[targets[:, 0] < width]
            targets = targets[targets[:, 2] > 0]
            targets = targets[targets[:, 1] < height]
            targets = targets[targets[:, 3] > 0]
        new_targets_all.append(targets)
    targets_all = new_targets_all
    return imgs, targets_all


def _mirror(image, boxes, flip_flag):
    _, width, _ = image.shape
    if flip_flag:
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


class MotTrainTransform(TrainTransform):

    def __call__(self, images, targets, input_dim):
        flip_flag = True if random.randrange(2) else False
        output_img = [] 
        output_label = []
        for image, target in zip(images, targets):
            image_t, padded_labels = self.core(image, target, input_dim, flip_flag)
            output_img.append(image_t)
            output_label.append(padded_labels)
        return output_img, output_label

    def core(self, image, targets, input_dim, flip_flag):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        ids = targets[:, 5].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        ids_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        image_t = _distort(image)
        image_t, boxes = _mirror(image_t, boxes, flip_flag)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        ids_t = ids[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            ids_t = ids_o

        labels_t = np.expand_dims(labels_t, 1)
        ids_t = np.expand_dims(ids_t, 1)

        targets_t = np.hstack((labels_t, boxes_t, ids_t))
        padded_labels = np.zeros((self.max_labels, 6))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels
    


class RandomErasing(object):
    def __init__(self, p=0.5, area_keep=0.7, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=114, inplace=True):
        super().__init__()
        logger = logging.getLogger(__name__)
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            logger.warning("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.area_keep = area_keep
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value) :
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_h, img_w, img_c = img.shape
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([h, w, img_c], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[None, None, :]

            i = torch.randint(0, img_h + 1, size=(1,)).item() - h // 2
            j = torch.randint(0, img_w + 1, size=(1,)).item() - w // 2
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, None
    
    @staticmethod
    def erase(img, y, x, h, w, v):
        img[y:y+h,x:x+w] = v
    
    @staticmethod
    def box_area(boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_overlap_select(self, boxes1, boxes2, img_h, img_w):
        if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
            return boxes1
        boxes1[0::2] = np.clip(boxes1[0::2], a_min=0, a_max=img_w)
        boxes1[1::2] = np.clip(boxes1[1::2], a_min=0, a_max=img_h)
        area2 = self.box_area(boxes2)

        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt)  # [N,M,2]
        wh[wh<0] = 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        overlap = inter / (area2[None, :] + 1e-6)
        assert overlap.shape[0] == 1
        select_idx = overlap.reshape(-1) < self.area_keep
        return select_idx
        
    def erase_img(self, img, targets_raw, targets, value):
        img_h, img_w, img_c = img.shape
        for _ in range(100):
            y, x, h, w, v = self.get_params(img, self.scale, self.ratio, value)
            if v is None:
                continue
            box1 = np.array([x, y, x+w, y+h]).reshape(1, 4)
            select_idx = self.box_overlap_select(box1, targets, img_h=img_h, img_w=img_w)
            targets_select = targets_raw[select_idx]
            if targets_select.shape[0] > 0:
                for x1, y1, x2, y2 in box1:
                    self.erase(img, y1, x1, y2-y1, x2-x1, value)
                return img, targets_select
        return img, targets_raw
    
    def core(self, img_all, label_raw_all, value):
        img_res, label_res = [], []
        for img, label_raw in zip(img_all, label_raw_all):
            if torch.rand(1) < self.p:
                label = deepcopy(label_raw)
                label = label.astype(np.int32)[:, :4]
                img_h, img_w, img_c = img.shape
                label[0::2] = np.clip(label[0::2], a_min=0, a_max=img_w)
                label[1::2] = np.clip(label[1::2], a_min=0, a_max=img_h)
                img, label_selected = self.erase_img(img, deepcopy(label_raw), label, value)
                label_res.append(label_selected)
            else:
                label_res.append(label_raw)
            img_res.append(img)
        return img_res, label_res

            
    def __call__(self, img, label):
        if isinstance(self.value, (int, float)):
            value = [self.value]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, tuple):
            value = list(self.value)
        else:
            value = self.value

        if value is not None and not (len(value) in (1, img[0].shape[-3])):
            raise ValueError(
                "If value is a sequence, it should have either a single value or "
                f"{img[0].shape[-3]} (number of input channels)"
            )
        if self.p > 0:
            img, label = self.core(img, label, value)

        return img, label

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}"
            f"(p={self.p}, "
            f"scale={self.scale}, "
            f"ratio={self.ratio}, "
            f"value={self.value}, "
            f"inplace={self.inplace})"
        )
        return s
