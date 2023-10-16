# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from motlib.utils import Registry


MOTDATASETFILE = Registry("MOTDATASETFILE")

from .mot17 import *
from .mot20 import *
from .crowdhuman import *
from .bdd100k import *
from .dancetrack import *
from .inference import *


def get_datasetfile(name, *args, **kwargs):
    return MOTDATASETFILE.get(name)(*args, **kwargs)