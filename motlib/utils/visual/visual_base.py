from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from pathlib import Path


class VisualBase(object):
    def __init__(self, cfg, start_epoch, rank):
        self.rank = rank
        self.start_epoch = start_epoch

        self._env_dir = Path(cfg.visual_env_dir)
        self._env_dir.mkdir(parents=True, exist_ok=True)
        self._env_name = cfg.job_name
        self._output_dir = Path(cfg.output_dir) / 'vis'
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._port = cfg.visual_port 
        self._host = cfg.visual_host
        self._copy = cfg.visual_copy

        self.prefix = ''
        self.postfix = ''

        self.init()

    def flush(self):
        raise NotImplementedError

    def has_writer(self):
        return False if self._writer is None else True

    def init(self):
        raise NotImplementedError

    def _legend_wrap(self, name):
        prefix = self.prefix + '_' if self.prefix != '' else ''
        postfix = '_' + self.postfix if self.postfix != '' else ''
        if self.rank > 0:
            return prefix + 'DR{}_'.format(self.rank) + name + postfix
        else:
            return prefix + name + postfix

    def close(self):
        pass