# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from motlib import mot_dataset

from .coco import build as build_coco
from motlib.mot_dataset.data_manager.tbd import build as build_tbd
from motlib.mot_dataset.data_manager.mot2coco import MOT2CoCoDataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, (torchvision.datasets.CocoDetection, MOT2CoCoDataset)):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    elif args.dataset_setting == 'tbd':
        return build_tbd(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
