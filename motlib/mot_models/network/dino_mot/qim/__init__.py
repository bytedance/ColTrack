from motlib.utils import Registry

QIM_REGISTRY = Registry("QIM")

from .motr import *
from .time_track import *


def build_qim(name, *args, **kwargs):
    return QIM_REGISTRY.get(name)(*args, **kwargs)
