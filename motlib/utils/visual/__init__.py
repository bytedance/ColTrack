from .visdom import VisualVisdom
from .tensorboard import VisualTensorboard
from motlib.utils import AverageMeter
from collections import OrderedDict
import numpy as np
import logging
import torch
import util.misc as comm


__all__ = ['VisStorage']


class VisStorage(object):
    def __init__(self, cfg, start_epoch=0, rank=None):
        platform = {
            'Visdom': VisualVisdom,
            'TensorBoard': VisualTensorboard
        }[cfg.visual_platform]

        if rank is None:
            rank = comm.get_rank()
        self.rank = rank
        self.board = platform(cfg, start_epoch, self.rank)
        self.finish = self.board.close

        self._copy = cfg.visual_copy
        self._flush_epoch = cfg.visual_flush_epoch 
        self._flush_id = 0

        self._bank = OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    @property
    def prefix(self, value):
        if hasattr(self.board, 'prefix'):
            self.board.prefix = value

    @property
    def postfix(self, value):
        if hasattr(self.board, 'postfix'):
            self.board.postfix = value

    def display(self, step):
        self._flush_id += 1

        for win in self._bank.keys():
            bank = self._bank[win]

            display_info = {}
            print_info = 'Epoch {}'.format(step)
            for k in bank.keys():
                info = bank[k]
                for name, value in info:
                    if win == 'terminal':
                        print_info += ' {} {}'.format(name, value)
                    else:
                        display_info[name] = value
                info.reset()
            if win == 'terminal':
                logger = logging.getLogger(__name__)
                logger.info(print_info)
            else:
                self.plot('add_scalars', win, display_info, step)

    def record(self, win_name: str, info: dict):
        if win_name not in self._bank:
            self._bank[win_name] = {}
        bank = self._bank[win_name]

        assert isinstance(info, dict), type(info)
        info_keys = list(info.keys())
        n = len(info_keys)
        m = len(list(bank.keys()))

        find_idx = np.zeros(n)

        for i_k, k in enumerate(info_keys):
            for idx in bank.keys():
                if k in bank[idx].name:
                    find_idx[i_k] = idx

        info_idx = int(find_idx[0])
        if find_idx.sum() == 0:
            bank[m+1] = AverageMeter(info_keys)
            info_idx = m + 1

        if np.unique(find_idx).size != 1:
            raise ValueError('Information is not recorded on one AverageMeter for ' + str(info_keys))

        values = []
        for i, k in enumerate(info_keys):
            values.append(info[k])
            assert k == bank[info_idx].name[i]
        bank[info_idx].update(values)

    def plot(self, func_name, *args, **kwargs):
        if self.board.has_writer():
            try:
                func = getattr(self.board, func_name)
                func(*args, **kwargs)

            except (ConnectionError, ConnectionRefusedError, ImportError) as e:
                logger = logging.getLogger(__name__)
                logger.error(e)
                logger.error('Web visualization is not available.')
            else:
                if self._copy and self.rank == 0:
                    if self._flush_id % self._flush_epoch == 0:
                        with torch.cuda.stream(torch.cuda.Stream()):
                            self.board.flush()