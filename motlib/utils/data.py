import torch
import numpy as np
from pathlib import Path
import torch
from collections import defaultdict


__all__ = ['to_cpu', 'load_dino_results']

def to_cpu(data):
    if isinstance(data, (int, float, np.ndarray)):
        return data
    elif isinstance(data, (list, tuple)):
        return [to_cpu(d) for d in data]
    elif isinstance(data, dict):
        return {k: to_cpu(v) for k,v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()
        return data.numpy()
    else:
        raise TypeError('not implement for type {}'.format(type(data)))


def load_dino_results(result_folder):
    result_folder = Path(result_folder)
    result_files = result_folder.glob("results-*.pkl")
    all_state_dict = defaultdict(list)
    for result_file in sorted(result_files):
        state_dict=torch.load(str(result_file))
        for k, v in state_dict.items():
            all_state_dict[k].extend(v)
    res = {}
    img_id_set = []
    cnt = 0
    for idx in range(len(all_state_dict['target'])):
        target = all_state_dict['target'][idx]
        result = all_state_dict['result'][idx]
        _image_id = int(target[0])
        _frame_id = int(target[1])
        _video_id = int(target[2])
       
        if _image_id in img_id_set:
            continue
        img_id_set.append(_image_id)

        res[_image_id] = (target, result)
        cnt += 1
    return res
