# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from . import DATALOADER_REGISTRY

import util.misc as utils
from motlib.mot_dataset.transform.yolox.data_augment import TrainTransform
from motlib.mot_dataset.transform.yolox.dataset import YOLODataset
from motlib.mot_dataset.transform.yolox.mosaic import MosaicDetection
from motlib.mot_dataset.transform.yolox.dataloader import YoloBatchSampler, DataLoader, InfiniteSampler
from torch.utils.data import DistributedSampler, RandomSampler


__alll__ = ['yolox_train_dataloader']


@DATALOADER_REGISTRY.register()
def yolox_train_dataloader(args):
    input_size = (800, 1440)
    args.img_size = input_size

    try:
        no_aug = not args.mosaic
    except:
        no_aug = False
        args.mosaic = True

    dataset = YOLODataset(
        args, train_or_test="train",
        transforms=TrainTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_labels=500,
        )
    )

    dataset = MosaicDetection(
            dataset,
            mosaic= not no_aug,
            img_size=input_size,
            args=args,
            train_or_test="train",
            transforms=TrainTransform(
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
