# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from pathlib import Path
import os
import sys
import json
import cv2
import glob as gb
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil
import torch
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from motlib.utils.file import set_dir
from motlib.utils.video_tools.util import draw_box
from motlib.utils.video_tools.img2video import img2video


def load_dict(json_dir):
    if isinstance(json_dir, dict):
        data = json_dir
    else:
        with open(str(json_dir), 'r') as f:
            data = json.load(f)

    video_name_dict = {}
    video_dict = {}
    for video_info in data['videos']:
        video_name_dict[video_info['id']] = video_info['name']
        video_dict[video_info['name']] = {}
    
    for img_info in data['images']:
        video_id = img_info['video_id']
        frame_id = img_info['frame_id']
        video_dict[video_name_dict[video_id]][frame_id] = img_info['file_name']
    return video_dict, video_name_dict

def det2img(result_path, json_path, visual_path):
    videos_dict, video_name_dict = load_dict(json_path)
    result_path = Path(result_path)
    visual_path = Path(visual_path)
    set_dir(visual_path)

    result_files = result_path.glob("results-*.pkl")
    all_state_dict = {}
    for result_file in sorted(result_files):
        state_dict=torch.load(str(result_file))
        for k, v in state_dict.items():
            if k not in all_state_dict:
                all_state_dict[k] = v
            else:
                all_state_dict[k].extend(v)
    
    video_result = {}
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
        if _video_id not in video_result:
            video_result[_video_id] = {}
        video_result[_video_id][_frame_id] = {'target': target, 'result': result}
        cnt += 1
    cnt = 0
    for video_id in sorted(list(video_result.keys())):
        video_name = video_name_dict[video_id]
        video_info = videos_dict[video_name]
        video_result_i  = video_result[video_id]
        for frame_id in sorted(list(video_result_i.keys())):
            img_dir = video_info[frame_id]
            boxes = video_result_i[frame_id]['result'][:, :4]
            scores = video_result_i[frame_id]['result'][:, 4]
            labels = video_result_i[frame_id]['result'][:, 5]
            select = scores > 0.2
            boxes = boxes[select]
            labels = labels[select]
            scores = scores[select]

            img = draw_box(img_dir, box_info=boxes, pids=None, scores=scores, classes=labels)
            img_out_dir = str(visual_path / "{:0>6d}.png".format(cnt))
            cv2.imwrite(img_out_dir, img)
            cnt += 1
            if cnt % 20 == 0:
                print('processed {} frames'.format(cnt))
        print(video_name, "Done")
    print('all done')


def txt2img(result_path, json_path, visual_path):
    videos_dict, _ = load_dict(json_path)
    result_path = Path(result_path)
    visual_path = Path(visual_path)
    set_dir(visual_path)
    result_dirs = result_path.glob("*.txt")

    cnt = 0
    for txt_path in sorted(result_dirs):
        video_name = str(txt_path.stem)
        txt_dict = defaultdict(list)   
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')

                pid = int(linelist[7])
                tid = int(linelist[1])
                img_id = int(linelist[0])
                score = float(linelist[6])
                class_id = float(linelist[7])

                bbox = [float(linelist[2]), float(linelist[3]), 
                        float(linelist[2]) + float(linelist[4]), 
                        float(linelist[3]) + float(linelist[5]), tid, score, class_id]
                txt_dict[int(img_id)].append(bbox)

        video_dict = videos_dict[video_name]
        for img_id in sorted(txt_dict.keys()):
            img_dir = video_dict[img_id]

            track_info = np.asarray(txt_dict[img_id])

            img = draw_box(str(img_dir), box_info=track_info[:, :4], pids=track_info[:, 4], scores=track_info[:, 5], classes=track_info[:, 6])

            img_out_dir = str(visual_path / "{:0>6d}.png".format(cnt))
            cv2.imwrite(img_out_dir, img)
            cnt += 1
            if cnt % 20 == 0:
                print('processed {} frames'.format(cnt))
        print(video_name, "Done")
    print(f'Txt2images done. The video frames drawn from the tracking results are stored in {str(visual_path)}.')


if __name__ == '__main__':
    track_result_dir = ''
    json_dir = ''
    frame_dir = ''
    video_dir = ''
    
    txt2img(track_result_dir, json_dir, frame_dir)

    img2video(frame_dir, video_dir)




