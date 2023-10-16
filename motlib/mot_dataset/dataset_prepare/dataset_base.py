# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from pathlib import Path
from motlib.utils import torch_distributed_zero_first
import util.misc as utils
import json
import motlib.evaluation.evaluate_metrics.trackeval_manager.trackeval as trackeval
from motlib.evaluation.evaluate_metrics.trackeval_manager.dataset import get_eval_dataset_func


class DatasetFileBase(object):
    def __init__(self, data_root, info_root):
        self.data_root = Path(data_root) if Path(data_root).is_absolute() else Path(data_root).resolve()

        if info_root is not None:
            self.info_root = Path(info_root) if Path(info_root).is_absolute() else Path(info_root).resolve()
            self.meta_dir = self.info_root / 'meta_file'
            self.meta_dir.mkdir(parents=True, exist_ok=True)
            self.meta_file = self.meta_dir / '{}.json'.format(self.__class__.__name__)
        else:
            self.meta_file = None

        self.init()
        self.eval_config = self._get_eval_setting()
        self.split_dict = self._get_split_dict()
        self.get_category_map()
        assert self.data_path.is_absolute()
    
    def add_subdirectory(self, data):

        for _, split in data.items():
            for video in split['videos']:
                for _, frame in video['frames'].items():
                    raw_dir = Path(frame['dir'])
                    if not raw_dir.is_absolute():
                        whole_dir = self.data_path / str(raw_dir)
                        new_dir = str(whole_dir)[len(str(self.data_root)) +1:]
                        frame['dir'] = new_dir
        return data   

    
    def init(self):
        self.category = [{'supercategory': 'none', 'id': 1, 'name': 'pedestrian'}]
    
    def get_category_map(self):
        category2id_map = {}
        id2category_map = {}
        for cls in self.category:
            category2id_map[cls['name']] = cls['id']
            id2category_map[int(cls['id'])] = cls['name']
        self.category2id_map = category2id_map
        self.id2category_map = id2category_map
    
    def _get_eval_setting(self):
        eval_dataset_class = 'MyMotChallenge2DBox'
        eval_dataset_func = get_eval_dataset_func(eval_dataset_class)
        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config['DISPLAY_LESS_PROGRESS'] = False
        default_dataset_config = eval_dataset_func.get_default_dataset_config()
        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

        config = {
            'name': 'Default',
            'evaluator': default_eval_config,
            'dataset': default_dataset_config,
            'metrics': default_metrics_config,
            'dataset_class': eval_dataset_class,
            'write_gt_func': 'write_gt_motchallenge'
            }

        return config

    def load(self):
        raise NotImplementedError
    
    def _load_after_post_process(self):
        res = self.load()
        res = self.add_subdirectory(res)
        for _, v in res.items():
            if 'category' not in v:
                v['category'] = self.category
            if 'eval_config' not in v:
                v['eval_config'] = self.eval_config
        return res
    
    def check_meta_file(self):
        if not self.meta_file.exists():
            output = self._load_after_post_process()
            with open(str(self.meta_file), 'w') as f:
                    json.dump(output, f)
    
    def _get_split_dict(self):
        if self.meta_file is not None:
            if utils.is_dist_avail_and_initialized():
                with torch_distributed_zero_first(utils.get_rank()):
                    self.check_meta_file()
            else:
                self.check_meta_file()
                
            with open(str(self.meta_file), 'r') as f:
                output = json.load(f)
        else:
            output = self._load_after_post_process()
        return output