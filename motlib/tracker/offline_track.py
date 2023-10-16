import torch
from pathlib import Path
import shutil
import logging
import os
from multiprocessing.pool import Pool
from functools import partial
from motlib.tracker import build_tracker
from motlib.utils import set_dir


def track_core(output, args, video_info):
    tracker = build_tracker(args.tracker_name, args, video_info, args.track_results_path)
    for idx, frame_id in enumerate(sorted(list(output.keys()))):
        target = {
            'image_id': output[frame_id]['target'][0],
            'frame_id': output[frame_id]['target'][1],
            'video_id': output[frame_id]['target'][2]
            }
        result = {
            'boxes': output[frame_id]['result'][:, :4],
            'scores': output[frame_id]['result'][:, 4],
            'labels': output[frame_id]['result'][:, 5]
            }
        tracker.track(target, result, idx == len(output)-1)


def track_by_offline(args, video_info, result_folder, num_imgs):
    args.track_results_path = '{}/{}/track_results'.format(args.output_dir, args.tracker_name)
    set_dir(args.track_results_path )

    result_folder = Path(result_folder)
    result_files = result_folder.glob("results-*.pkl")
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
    assert cnt == num_imgs, '{} and {}'.format(cnt, num_imgs)
    
    if args.USE_PARALLEL:
        with Pool(args.NUM_PARALLEL_CORES) as pool:
            video_result_list = []
            for video_id in sorted(list(video_result.keys())):
                video_result_list.append(video_result[video_id])

            _track_core = partial(track_core, args=args, video_info=video_info)
            pool.map(_track_core, video_result_list)
    else:
        for video_id in sorted(list(video_result.keys())):
            track_core(video_result[video_id], args, video_info)
    


