# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.mot_dataset.dataset_prepare import MOTDATASETFILE
from .mot17 import MOT17


__all__ = ['MOT20']


@MOTDATASETFILE.register()
class MOT20(MOT17):
    def init(self):
        super().init()
        self.subdirectory = 'MOT20/images'
        self.data_path = self.data_root / self.subdirectory
        self.name_key = "MOT20-*"

