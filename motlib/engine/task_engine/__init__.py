from motlib.utils import Registry

ENGIN_REGISTRY = Registry("ENGIN")

from .default_engine import *
from .mot_engine import *


def build_engin(name, *args, **kwargs):
    return ENGIN_REGISTRY.get(name)(*args, **kwargs)