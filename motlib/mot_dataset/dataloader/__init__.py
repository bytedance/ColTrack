from motlib.utils import Registry


DATALOADER_REGISTRY = Registry("DATALOADER")

from .default_dataloader import *
from .yolox_dataloader import *
from .mot_dataloader import *

def get_dataloader(name, *args, **kwargs):
    return DATALOADER_REGISTRY.get(name)(*args, **kwargs)

