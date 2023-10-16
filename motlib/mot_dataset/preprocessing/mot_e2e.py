# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from copy import deepcopy
import util.misc as utils
from motlib.mot_models.structures import Instances

import torch
from functools import partial
import logging


def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets


def tensor_to_cuda(tensor: torch.Tensor, device):
    return tensor.to(device)


def is_tensor_or_instances(data):
    return isinstance(data, torch.Tensor) or isinstance(data, Instances)


def targets_to_instances(targets, img_shape):
    gt_instances = Instances(tuple(img_shape))
    assert targets.shape[0] == 1, 'Shape {}'.format(str(targets.shape))
    targets = targets[0]
    h, w = tuple(img_shape)
    targets = targets[targets.sum(axis=1) > 0]
    # cxcywh
    boxes = targets[:, 1:5] / torch.tensor([w, h, w, h], dtype=torch.float32)
    labels = targets[:, 0].long()
    obj_ids = targets[:, 5].long()
    target_output = {"boxes": boxes, "labels": labels}
    gt_instances.boxes = deepcopy(boxes)
    gt_instances.labels = deepcopy(labels)
    gt_instances.obj_ids = deepcopy(obj_ids)
    # gt_instances.area = targets['area']
    return gt_instances, target_output


def data_apply(data, check_func, apply_func):
    if isinstance(data, dict):
        for k in data.keys():
            if check_func(data[k]):
                data[k] = apply_func(data[k])
            elif isinstance(data[k], dict) or isinstance(data[k], list):
                data_apply(data[k], check_func, apply_func)
            else:
                raise ValueError()
    elif isinstance(data, list):
        for i in range(len(data)):
            if check_func(data[i]):
                data[i] = apply_func(data[i])
            elif isinstance(data[i], dict) or isinstance(data[i], list):
                data_apply(data[i], check_func, apply_func)
            else:
                raise ValueError("invalid type {}".format(type(data[i])))
    elif check_func(data):
        data = apply_func(data)
    else:
        raise ValueError("invalid type {}".format(type(data)))
    return data


def data_dict_to_cuda(data_dict, device):
    return data_apply(data_dict, is_tensor_or_instances, partial(tensor_to_cuda, device=device))


def data_preprocess_e2e(device, samples_all, targets_all, img_info_all=None, image_id_all=None):
    # height, width, frame_id, video_id, file_name = img_info
    if isinstance(samples_all, (list, tuple)):
        gt_instances = []
        targets_output = []
        for samples, targets in zip(samples_all, targets_all):
            gt_instances_i, targets_i = targets_to_instances(targets, samples.shape[2:4]) 
            gt_instances.append(gt_instances_i)
            targets_output.append(targets_i)
        
        samples_all = torch.cat(samples_all, dim=0)
        samples_all.requires_grad = False
        samples_all = utils.nested_tensor_from_tensor_list(samples_all.to(device))

        gt_instances = data_dict_to_cuda(gt_instances, device)
        targets_output = data_dict_to_cuda(targets_output, device)
        return samples_all, targets_output, gt_instances
    else:
        raise NotImplementedError