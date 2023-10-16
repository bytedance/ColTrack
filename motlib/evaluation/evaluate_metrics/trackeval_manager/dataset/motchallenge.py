from motlib.evaluation.evaluate_metrics.trackeval_manager.trackeval.datasets import MotChallenge2DBox

import os
import json
from . import EVALDATASET


__all__ = ['MyMotChallenge2DBox']


@EVALDATASET.register()
class MyMotChallenge2DBox(MotChallenge2DBox):
    def _get_seq_info(self):
        seq_info_dir = self.config['SEQMAP_FILE']
        assert os.path.exists(seq_info_dir)
        with open(seq_info_dir, 'r') as f:
            seq_lengths = json.load(f)
        return sorted(list(seq_lengths.keys())), seq_lengths