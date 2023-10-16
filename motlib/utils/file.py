import numpy as np
from pathlib import Path
import time
import util.misc as utils
import json
import os
import shutil
import logging

__all__ = ['wait_for_load', 'write_json_file', 'set_dir']


def wait_for_load(file_dir):
    file_dir = Path(file_dir)
    cnt = 0
    max_time = 1200
    while cnt < max_time:
        if not file_dir.exists():
            time.sleep(1)
            cnt += 1
        else:
            break
    if cnt >=max_time:
        raise TimeoutError('data json timeout ofr {}'.format(str(file_dir)))


def write_json_file(file_dir, data, local_rank=0):
    file_dir = Path(file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    if not file_dir.exists():
        if (utils.is_dist_avail_and_initialized() and local_rank == 0) or not utils.is_dist_avail_and_initialized():
            with open(str(file_dir), 'w') as f:
                json.dump(data, f)
        else:
            wait_for_load(file_dir)


def set_dir(file_dir):
    if os.path.exists(file_dir):
        try:
            shutil.rmtree(file_dir)
        except OSError as e:
            logger = logging.getLogger(__name__)
            logger.error("Error: %s - %s." % (e.filename, e.strerror)) 
    os.makedirs(file_dir, exist_ok=True)