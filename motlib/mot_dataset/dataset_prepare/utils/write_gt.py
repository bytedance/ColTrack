import json
from pathlib import Path
import numpy as np
from copy import deepcopy


def write_gt(data_root, gt_path, raw_videos_info):
    if raw_videos_info['eval_config'] is None:
        return
    write_gt_func_name = raw_videos_info['eval_config']['write_gt_func']
    if write_gt_func_name is None:
        return
    if write_gt_func_name == 'write_gt_motchallenge':
        write_gt_motchallenge(data_root, gt_path, raw_videos_info)
    elif write_gt_func_name == 'write_gt_bdd100k':
        write_gt_bdd100k(data_root, gt_path, raw_videos_info)
    else:
        raise KeyError


def write_gt_motchallenge(data_root, gt_path, raw_videos_info):
    gt_path = Path(gt_path)
    file_path = gt_path / 'gt'
    file_path.mkdir(parents=True, exist_ok=True)
    seq_info_dir = gt_path / 'seq_info.json'

    seq_info = {}
    for video_info in raw_videos_info['data']:
        video_name = video_info['name']
        H, W = video_info['height'], video_info['width']
        frame_num = video_info['frame_num']
        seq_info[video_name] = int(frame_num)
        file_dir = file_path / '{}.txt'.format(video_name)
        if file_dir.exists():
            continue

        with open(str(file_dir), 'w') as f:
            for k, v in video_info['frames'].items():
                frame_id = int(k)

                if 'boxs' in v:
                    raw_boxs = np.asarray(v['boxs'], np.float64)
                    boxs = raw_boxs.copy()
                    # conver cx, cy, w, h to lx, ly, rx, ry
                    boxs[:, 2] = W * (raw_boxs[:, 2] - raw_boxs[:, 4] / 2)
                    boxs[:, 3] = H * (raw_boxs[:, 3] - raw_boxs[:, 5] / 2)
                    boxs[:, 4] = W * raw_boxs[:, 4]
                    boxs[:, 5] = H * raw_boxs[:, 5]
                    for box in boxs:
                        f.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                            int(frame_id), int(box[0]), int(box[2]), int(box[3]), int(box[4]), int(box[5]), int(1), int(1), -1))
    if not seq_info_dir.exists():
        with open(str(seq_info_dir), 'w') as f:
            json.dump(seq_info, f)


def load_bdd100k_anns(anns_path):
    anns_path = Path(anns_path)
    anns_map ={}
    video_set = set()
    for anns_dir in sorted(anns_path.glob('*.json')):
        json_video_name = anns_dir.stem
        assert json_video_name not in video_set
        video_set.add(json_video_name)
        video_map = {}

        with open(str(anns_dir), 'r') as f:
            labels_json = json.load(f)
        for label_json in labels_json:
            img_name = label_json['name']
            video_map[img_name] = label_json
        anns_map[json_video_name] = video_map
    return anns_map


def write_gt_bdd100k(data_root, gt_path, raw_videos_info):
    gt_path = Path(gt_path)
    file_path = gt_path / 'gt'
    file_path.mkdir(parents=True, exist_ok=True)

    ann_path = Path(data_root) / raw_videos_info['eval_config']['anns_path']
    anns_map = load_bdd100k_anns(ann_path)

    for video_info in raw_videos_info['data']:
        video_name = video_info['name']
        # H, W = video_info['height'], video_info['width']
        file_dir = file_path / '{}.json'.format(video_name)
        if file_dir.exists():
            continue

        with open(str(file_dir), 'w') as f:
            videos_anns = []

            for k, v in video_info['frames'].items():
                img_name = Path(v['dir']).name
                if video_name in anns_map and img_name in anns_map[video_name]:
                    img_anns = deepcopy(anns_map[video_name][img_name])
                    img_anns['frameIndex'] = int(k) - 1

                    videos_anns.append(img_anns)
            json.dump(videos_anns, f)

