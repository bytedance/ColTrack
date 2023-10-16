# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import torch
import util.misc as utils


def data_preprocess_tbd(samples, targets, img_info, image_id):
    # height, width, frame_id, video_id, file_name = img_info
    h = samples.shape[2]
    w = samples.shape[3]
    samples = utils.nested_tensor_from_tensor_list(samples)
    target_output = []
    for idx, t_info in enumerate(targets):
        t_info = t_info[t_info.sum(axis=1) > 0]
        
        target = {}
        target["boxes"] = t_info[:, 1:5] / torch.tensor([w, h, w, h], dtype=torch.float32)
        target["labels"] = t_info[:, 0].long()
        target["image_id"] = torch.tensor([image_id[idx]])
        target['track_id'] = t_info[:, 5]
        target_output.append(target)

    return samples, target_output