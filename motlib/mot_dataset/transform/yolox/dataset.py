from motlib.mot_dataset.data_manager.mot2coco import MOT2CoCoDataset

import json
import os
import torch
from pathlib import Path
import numpy as np
from functools import wraps
import cv2


class Dataset(MOT2CoCoDataset):
    def __init__(self, args, train_or_test, transforms=None) -> None:
        super().__init__(args, train_or_test, transforms)

        self.__input_dim = args.img_size[:2]
        self.enable_mosaic = args.mosaic
    
    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim
    
    @staticmethod
    def resize_getitem(getitem_fn):
        """
        Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the on the fly resizing of
        the ``input_dim`` with our :class:`~lightnet.data.DataLoader` class.

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.resize_getitem
            ...     def __getitem__(self, index):
            ...         # Should return (image, anno) but here we return input_dim
            ...         return self.input_dim
            >>> data = CustomSet((200,200))
            >>> data[0]
            (200, 200)
            >>> data[(480,320), 0]
            (480, 320)
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                has_dim = True
                self._input_dim = index[0]
                self.enable_mosaic = index[2]
                index = index[1]
            else:
                has_dim = False

            ret_val = getitem_fn(self, index)

            if has_dim:
                del self._input_dim

            return ret_val

        return wrapper
    

class YOLODataset(Dataset):
    def __init__(self, args, train_or_test, transforms=None) -> None:
        super().__init__(args, train_or_test, transforms)
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.img_size = args.img_size
        
    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = obj["category_id"]
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"]
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]

        img = cv2.imread(file_name)
        assert img is not None

        return img, res.copy(), img_info, np.array([id_])
    
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
        img, target, img_info, img_id = self.pull_item(index)

        if self._transforms is not None:
            img, target = self._transforms(img, target, self.input_dim)
        return img, target, img_info, img_id
    
    