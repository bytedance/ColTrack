import torch
from contextlib import contextmanager
import util.misc as utils

  
__all__ = ['torch_distributed_zero_first']


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        if utils.is_dist_avail_and_initialized():
            torch.distributed.barrier()
    yield   
    if local_rank == 0:
        if utils.is_dist_avail_and_initialized():
            torch.distributed.barrier()