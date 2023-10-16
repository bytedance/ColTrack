# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from . import DATALOADER_REGISTRY

import torch
import util.misc as utils
from motlib.mot_dataset.transform.mot_video.data_augment import MotTrainTransform
from motlib.mot_dataset.transform.mot_video.dataset import MOTYOLODataset
from motlib.mot_dataset.transform.mot_video.mosaic import MOTMosaicDetection

from motlib.mot_dataset.transform.yolox.dataloader import YoloBatchSampler, DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from motlib.mot_dataset.sampler.e2etest_distributed import MOTE2ETestDistributedSampler

from motlib.mot_dataset.data_manager.tbd import build as build_tbd


__alll__ = ['mot_train_dataloader', 'mot_test_dataloader']


@DATALOADER_REGISTRY.register()
def mot_train_dataloader(args):
    input_size = (800, 1440)
    args.img_size = input_size

    try:
        no_aug = not args.mosaic
    except:
        no_aug = False
        args.mosaic = True

    dataset = MOTYOLODataset(
        args, train_or_test="train",
        transforms=MotTrainTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_labels=500,
        )
    )

    dataset = MOTMosaicDetection(
            dataset,
            mosaic= not no_aug,
            img_size=input_size,
            args=args,
            train_or_test="train",
            transforms=MotTrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=10.0,
            translate=0.1,
            scale=(0.1, 2),
            shear=2.0,
            perspective=0.0,
            enable_mixup=True,
        )
    seed = args.seed + utils.get_rank()
    # sampler = InfiniteSampler(len(dataset), seed=seed)
    if args.distributed:
        sampler = DistributedSampler(dataset, seed=seed)
    else:
        sampler = RandomSampler(dataset)

    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=args.batch_size,
        drop_last=False,
        input_dimension=input_size,
        mosaic=not no_aug,
    )

    dataloader_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    train_loader = DataLoader(dataset, **dataloader_kwargs)
    return train_loader


@DATALOADER_REGISTRY.register()
def mot_test_dataloader(args):
    dataset_val = build_tbd(image_set='test', args=args)
    if args.distributed:
        sampler_val = MOTE2ETestDistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        raise NotImplementedError
    data_loader_val = torch.utils.data.DataLoader(dataset_val, args.test_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    return data_loader_val