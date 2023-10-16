from motlib.utils import Registry


NETWORK_REGISTRY = Registry("MOTNETWORK")

from .dino_mot import *

def build_network(name, *args, **kwargs):
    return NETWORK_REGISTRY.get(name)(*args, **kwargs)