# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from . import DATALOADER_REGISTRY

from datasets import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
import torch

__all__ = ['default_train_dataloader', "default_test_dataloader"]


@DATALOADER_REGISTRY.register()
def default_train_dataloader(args):
    dataset_train = build_dataset(image_set='train', args=args)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    return data_loader_train


@DATALOADER_REGISTRY.register()
def default_test_dataloader(args):
    dataset_val = build_dataset(image_set='test', args=args)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.test_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    return data_loader_val