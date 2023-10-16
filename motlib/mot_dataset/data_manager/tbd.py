# Copyright (2023) Bytedance Ltd. and/or its affiliates 



from datasets.coco import make_coco_transforms
# from motlib.mot_dataset.transform.tbd_transforms import make_coco_transforms_tbd as make_coco_transforms
from .mot2coco import MOT2CoCoDataset


def build(image_set, args):
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    dataset = MOT2CoCoDataset(
        args, image_set,
        transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args)
        )
    return dataset


