from pathlib import Path
import os
import sys
import json
import cv2
import glob as gb
import numpy as np
from pathlib import Path
from collections import defaultdict


def draw_box(img_dir_or_img, box_info, pids=None, scores=None, classes=None):
    if isinstance(img_dir_or_img, str):
        img = cv2.imread(img_dir_or_img)
    elif isinstance(img_dir_or_img, np.ndarray):
        img = img_dir_or_img
    else:
        raise TypeError

    box_info = np.asarray(box_info)
    if len(box_info) == 0:
        return img
    assert box_info.shape[1] == 4
    color_list = colormap()

    text_scale = max(0.5, img.shape[1] / 3200.)
    text_thickness = max(2, int(img.shape[1] / 800.))
    line_thickness = max(2, int(img.shape[1] / 700.))

    if pids is not None:
        pids = np.asarray(pids)
        pids.shape[0] == box_info.shape[0]
    if scores is not None:
        scores = np.asarray(scores)
        scores.shape[0] == box_info.shape[0]
    if classes is not None:
        classes = np.asarray(classes)
        classes.shape[0] == box_info.shape[0]

    for bid, bbox in enumerate(box_info):
        color_idx = int(bbox[3]%79)
        txt_info = ''

        if pids is not None:
            pid = int(pids[bid])
            color_idx = int(pid%79)
            txt_info += str(pid)
        if scores is not None:
            if len(txt_info) > 0:
                txt_info += '_'
            txt_info += '{:.1f}'.format(scores[bid]*100)
        if classes is not None:
            if len(txt_info) > 0:
                txt_info += '_'
            txt_info += '{}'.format(int(classes[bid]))
        
        if len(txt_info) == 0:
            txt_info = None

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[color_idx].tolist(), thickness=line_thickness)
        
        if txt_info is not None:
            cv2.putText(img, txt_info, (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color_list[color_idx].tolist(), text_thickness)
    return img


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list