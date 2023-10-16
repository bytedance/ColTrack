from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import logging
from PIL import Image
import json
from json import JSONDecodeError
from .visual_base import VisualBase
import util.misc as comm
from motlib.utils import PathManager
from torch.utils.tensorboard import SummaryWriter


class VisualTensorboard(VisualBase):
    def init(self):
        logger = logging.getLogger(__name__)
        self._writer = SummaryWriter(log_dir=str(self._output_dir), purge_step=self.start_epoch)
        logger.info('Tensorboard is available. Using tensorboard to display visual data on the web page.')

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def flush(self):
        pass

    def close(self):
        self.flush()