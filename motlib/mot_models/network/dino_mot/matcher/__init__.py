from motlib.utils import Registry


MOTMATCHER_REGISTRY = Registry("MOTMATCHER")

from .default_matcher import *


def build_matcher(name, *args, **kwargs):
    return MOTMATCHER_REGISTRY.get(name)(*args, **kwargs)