from motlib.utils import Registry

MOTMODEL_REGISTRY = Registry("MOTMODEL")

from .default_model import *
from .time_track import *

def build_mot_model(name, *args, **kwargs):
    return MOTMODEL_REGISTRY.get(name)(*args, **kwargs)