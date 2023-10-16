from motlib.utils import Registry


TRACKER_REGISTRY = Registry("TRACKER")

from .byte_tracker import *
from .bot_tracker import *

def build_tracker(name, *args, **kwargs):
    return TRACKER_REGISTRY.get(name)(*args, **kwargs)
