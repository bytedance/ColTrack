import json
from util.slconfig import SLConfig
from copy import deepcopy
from pathlib import Path
import os
import numpy as np
import logging
import util.misc as utils
from motlib.mot_dataset.dataset_prepare.utils.coco_format import MotCOCO
from motlib.utils import torch_distributed_zero_first
from motlib.mot_dataset.dataset_prepare import get_datasetfile
from motlib.mot_dataset.dataset_prepare.utils.write_gt import write_gt


def get_cfg_by_split_name(split_name):
    cur_dir = Path(os.path.realpath(__file__)).parent
    cfg = SLConfig.fromfile(str(cur_dir / '../../dataset_config/{}.py'.format(split_name)))
    cfg_dict = cfg._cfg_dict.to_dict()
    cfg_dict['split_name'] = split_name
    return cfg_dict


def relabel_map(id_list, offset=0):
    id_map = {}
    for idx, old_id in enumerate(sorted(list(id_list))):
        id_map[old_id] = idx + offset
    return id_map



def category_merge(category_split_list):
    category2id_map = {'pedestrian': {'supercategory': 'none', 'id': 1, 'name': 'pedestrian'}}
    id_num = 2
    for category in category_split_list:
        for category_i in category:
            if category_i['name'] not in category2id_map:
                category_i_tmp = deepcopy(category_i)
                category_i_tmp['id'] = id_num
                id_num += 1
    
                category2id_map[category_i['name']] = category_i_tmp
            else:
                assert category_i['supercategory'] == category2id_map[category_i['name']]['supercategory']
    category = [v for k, v in category2id_map.items()]
    return category


def merge_dataset_split(splits_data, train_or_test):
    output = []
    image_id = 0
    track_id_offset = 0
    person_id_offset = 0
    eval_config = None

    category_list = [split['category'] for split in splits_data]
    category = category_merge(category_list)
    category2id_map = {v['name']: v['id']  for v in category}

    category_set = set()

    for split in splits_data:
        videos = split['videos']
        eval_config_split = split['eval_config']
        category_split = split['category']
        category_split_map = { c['id']: c['name'] for c in category_split}

        if train_or_test == 'test':
            if eval_config is None:
                eval_config = deepcopy(eval_config_split)
            else:
                assert eval_config['name'] == eval_config_split['name']

        person_id_set = set()

        # get person id map
        if 'id_num' in split and split['id_num'] != -1:
            for video_info in videos:
                for k, v in video_info['frames'].items():
                    if 'boxs' not in v:
                        continue
                    boxs = v['boxs']
                    for box in boxs:
                        if int(box[1]) != -1:
                            person_id_set.add(int(box[1]))

        person_id_map = relabel_map(person_id_set, person_id_offset)
        person_id_offset += len(person_id_set)
        person_id_map[-1] = -1
    
        for video_info in videos:

            tmp_video_info = deepcopy(video_info)
            tmp_frames_info = tmp_video_info['frames']
            tmp_frames_info_int_key = {}
            for k, v in tmp_frames_info.items():
                tmp_frames_info_int_key[int(k)] = v
            tmp_video_info['frames'] = tmp_frames_info_int_key
            video_info = tmp_video_info

            new_video_info = deepcopy(video_info)
            frames_info = video_info['frames']
            frame_id_tmp = list(sorted(frames_info.keys()))
            frame_id_map = relabel_map(frame_id_tmp, offset=1)
            track_id_set = set()
            
            for k, v in frames_info.items():
                if 'boxs' not in v:
                    continue
                boxs = v['boxs']
                for box in boxs:
                    track_id_set.add(int(box[0]))
            
            track_id_map = relabel_map(track_id_set, track_id_offset)
            track_id_offset += len(track_id_set)

            new_frames_info = {}
            for old_frame_id in frame_id_tmp:
                new_frames_info[frame_id_map[old_frame_id]] = {'dir': frames_info[old_frame_id]['dir'], 'image_id': image_id}
                image_id += 1
                if 'boxs' not in frames_info[old_frame_id]:
                    continue
                new_boxs = []
                frame_id_set = set()
                for box in frames_info[old_frame_id]['boxs']:
                    if train_or_test == 'train':
                        category_set.add(category_split_map[int(box[6])])

                    new_boxs.append([track_id_map[int(box[0])], person_id_map[int(box[1])]] + box[2:6] + [category2id_map[category_split_map[int(box[6])]]])
                    assert track_id_map[int(box[0])] not in frame_id_set
                    frame_id_set.add(track_id_map[int(box[0])])
                new_frames_info[frame_id_map[old_frame_id]]['boxs'] = new_boxs
            new_video_info['frames'] = deepcopy(new_frames_info)
            output.append(new_video_info)
    if train_or_test == 'train' and len(category_set) != len(category):
        logger = logging.getLogger(__name__)
        empty_category = ''
        for k, _ in category2id_map.items():
            if k not in category_set:
                empty_category += str(k) + ' '
        logger.warning('catyegory ' + empty_category + "didn't show up in the training set")

    return {'data':output, 'category': category, 'eval_config': eval_config}
            

def get_dataset_data(cfg_dict):
    logger = logging.getLogger(__name__)

    info_root = cfg_dict['info_root']
    data_root = cfg_dict['data_root']

    def _load_split_info(data_list, train_or_test):
        splits_data = []
        for k, v in data_list.items():
            datafile = get_datasetfile(k, data_root=data_root, info_root=info_root)
            for split in v:
                logger.info('Loadding split {} from dataset {} for {}'.format(split, k, train_or_test))
                splits_data.append(datafile.split_dict[split])
                logger.info('dataset loaded')
        res = merge_dataset_split(splits_data, train_or_test)
        return res
    
    output = {}
    if 'train_dataset' in cfg_dict:
        output['train'] = _load_split_info(cfg_dict['train_dataset'], 'train')
    if 'test_dataset' in cfg_dict:
        output['test'] = _load_split_info(cfg_dict['test_dataset'], 'test')
    return output


def convert_coco_type(raw_videos_info, data_root):
    video_id = 0
    box_id = 0
    out = {'images': [], 'annotations': [], 'videos': []}
    out['categories'] = raw_videos_info['category']
    out['eval_config'] = raw_videos_info['eval_config']
    
    video_set = []

    for video_info in raw_videos_info['data']:
        H, W = video_info['height'], video_info['width']
        frame_rate = video_info['frame_rate']
        dataset_name = video_info['dataset']
        video_name = video_info['name']
        if video_id not in video_set:
            out['videos'].append({'frame_rate': frame_rate, 'dataset':dataset_name, 'height': H, 'width':W, 'name': video_name, 'id': video_id})
            video_set.append(video_id)
        
        for k, v in video_info['frames'].items():
            img_dir = Path(data_root) / v['dir']
            img_info = {'file_name': str(img_dir), 'id': int(v['image_id']), 'frame_id': int(k), 'video_id':video_id, 'height': H, 'width':W}
            assert img_info['frame_id'] > 0
            out['images'].append(img_info)

            if 'boxs' in v:
                if len(v['boxs']) == 0:
                    continue
                raw_boxs = np.asarray(v['boxs'], np.float64)
                boxs = raw_boxs.copy()
                # conver cx, cy, w, h to lx, ly, w, h
                boxs[:, 2] = W * (raw_boxs[:, 2] - raw_boxs[:, 4] / 2)
                boxs[:, 3] = H * (raw_boxs[:, 3] - raw_boxs[:, 5] / 2)
                boxs[:, 4] = W * raw_boxs[:, 4]
                boxs[:, 5] = H * raw_boxs[:, 5]

                # assert (boxs[:, 4:6] >= boxs[:, 2:4]).all()
                for label in boxs:
                    target_i = {}
                    target_i['bbox'] = label[2:6].tolist()
                    target_i['area'] = label[4] * label[5]
                    target_i['iscrowd'] = 0
                    target_i['category_id'] = int(label[6])
                    target_i['id'] = box_id
                    target_i['image_id'] = int(v['image_id'])
                    target_i['track_id'] = int(label[0])
                    target_i['instance_id'] = int(label[1])
                    box_id += 1
                    out['annotations'].append(target_i)
            
            
        video_id += 1
    return out
   

def load_info(args, split_name, train_or_test='train', data_type='coco'):
    if not args.just_inference_flag:
        cfg_dict = get_cfg_by_split_name(split_name)

        split_path = Path(cfg_dict['info_root']) / 'dataset_split' / str(split_name)
        split_path.mkdir(parents=True, exist_ok=True)
        split_json_dir = split_path / 'split' / '{}.json'.format(split_name)
        gt_path = split_path / 'gt_{}'.format(train_or_test)

        if not split_json_dir.exists():
            with torch_distributed_zero_first(utils.get_rank()):
                split_json_dir.parent.mkdir(parents=True, exist_ok=True)
                raw_videos_info = get_dataset_data(cfg_dict)
                with open(str(split_json_dir), 'w') as f:
                    json.dump(raw_videos_info, f)

        with open(str(split_json_dir), 'r') as f:
            raw_videos_info = json.load(f)[train_or_test]
            if not gt_path.exists():
                gt_path.mkdir(parents=True, exist_ok=True)
                with torch_distributed_zero_first(utils.get_rank()):
                    write_gt(cfg_dict['data_root'], gt_path, raw_videos_info)
        
        if train_or_test == 'train':
            dataset_name = list(cfg_dict['train_dataset'].keys())
        elif train_or_test == 'test':
            dataset_name = list(cfg_dict['test_dataset'].keys())

    else:
        cfg_dict = {}
        cfg_dict['data_root'] = args.data_root
        cfg_dict['info_root'] = None
        cfg_dict['num_classes'] = args.num_classes
        cfg_dict['test_dataset'] = {'Inference':[f'val_{args.inference_sampler_interval}']}
        cfg_dict['split_name'] = split_name

        dataset_name = 'Inference'
        raw_videos_info = get_dataset_data(cfg_dict)[train_or_test]

    if data_type == 'coco':
        
        if not args.just_inference_flag:
            coco_dir = split_path / 'coco'  / '{}_{}.json'.format(split_name, train_or_test)
            if not coco_dir.exists():
                coco_dir.parent.mkdir(parents=True, exist_ok=True)
                with torch_distributed_zero_first(utils.get_rank()):
                    output = convert_coco_type(raw_videos_info, cfg_dict['data_root'])
                    with open(str(coco_dir), 'w') as f:
                        json.dump(output, f)
            output = MotCOCO(str(coco_dir))
        else:
            coco_dir = None
            output = convert_coco_type(raw_videos_info, cfg_dict['data_root'])
            output = MotCOCO(output)
            gt_path = None
        output.num_classes = cfg_dict['num_classes']
        output.dataset_name = dataset_name
        if output.eval_config is None:
            output.eval_config = {}
    
        for k, v in output.eval_config.items():
            if 'path' in k:
                output.eval_config[k] = str(Path(cfg_dict['data_root']) / output.eval_config[k])
        
        output.eval_config['gt_path'] = str(gt_path)
        output.eval_config['coco_path'] = str(coco_dir)
        return output
    else:
        raise NotImplementedError
    
    
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
    output = load_info( None,'bdd100k_ablation', 'test')
    a= 1
