from motlib.utils import Registry
from motlib.evaluation.evaluate_metrics.trackeval_manager.trackeval import datasets


EVALDATASET = Registry("EVALDATASET")

from .motchallenge import *

def get_eval_dataset_func(name):
    if hasattr(datasets, name):
        return getattr(datasets, name)
    return EVALDATASET.get(name)