from copy import deepcopy
from motlib.mot_models.structures import Instances
from motlib.mot_models.network.dino_mot.tracker.util import Track
from motlib.tracker.byte_tracker.manager import write_results
from motlib.utils import set_dir

import numpy as np
from pathlib import Path
import torch


class TrackBase(object):
    def __init__(self, args) -> None:
        self.args = args
        # self.max_age = self.args.max_age
        # self.min_hits = self.args.min_hits
        # self.iou_threshold = self.args.iou_threshold 

        self.trackers = []
        self.frame_count = 0
        self.active_trackers = {}
        self.inactive_trackers = {}
        self.disappeared_tracks = []
    
    def _remove_track(self, slot_id):
        self.inactive_trackers.pop(slot_id)
        self.disappeared_tracks.append(slot_id)

    def clear_disappeared_track(self):
        self.disappeared_tracks = []

    def update(self, dt_instances: Instances):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        dt_idxes = set(dt_instances.obj_idxes.tolist())
        track_idxes = set(self.active_trackers.keys()).union(set(self.inactive_trackers.keys()))
        matched_idxes = dt_idxes.intersection(track_idxes)

        unmatched_tracker = track_idxes - matched_idxes
        for track_id in unmatched_tracker:
            # miss in this frame, move to inactive_trackers.
            if track_id in self.active_trackers:
                self.inactive_trackers[track_id] = self.active_trackers.pop(track_id)
            self.inactive_trackers[track_id].miss_one_frame()
            if self.inactive_trackers[track_id].miss > 10:
                self._remove_track(track_id)

        for i in range(len(dt_instances)):
            idx = dt_instances.obj_idxes[i].item()
            bbox = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
            label = dt_instances.labels[i].item()
            if label == 1:
                # get a positive track.
                if idx in self.inactive_trackers:
                    # set state of track active.
                    self.active_trackers[idx] = self.inactive_trackers.pop(idx)
                if idx not in self.active_trackers:
                    # create a new track.
                    self.active_trackers[idx] = Track(idx)
                self.active_trackers[idx].update(bbox)
            elif label == 0:
                # get an occluded track.
                if idx in self.active_trackers:
                    # set state of track inactive.
                    self.inactive_trackers[idx] = self.active_trackers.pop(idx)
                if idx not in self.inactive_trackers:
                    # It's strange to obtain a new occluded track.
                    # TODO: think more rational disposal.
                    self.inactive_trackers[idx] = Track(idx)
                self.inactive_trackers[idx].miss_one_frame()
                if self.inactive_trackers[idx].miss > 10:
                    self._remove_track(idx)

        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i].item()
            if label == 1:
                id = dt_instances.obj_idxes[i].item()
                box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
                ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))


class E2ETrackManager(object):
    def __init__(self, args, valid_frame_num, video_info):
        self.args = args
        self.video_info_map = self.get_video_name_map(video_info)

        self.result_folder = Path('{}/{}/track_results'.format(args.output_dir, args.tracker_name))
        set_dir(self.result_folder)

        self.prob_threshold = self.args.prob_threshold
        self.num_select = self.args.num_select
        self.area_threshold = 100

        # self.tracker = TrackBase(self.args)

        self.valid_frame_num = valid_frame_num
        self.num = 0

        self.total_dts = 0
        self.total_occlusion_dts = 0

        self.last_frame_id = 0
        self.last_video_id = -1
        self.results_track = []
    
    def get_video_name_map(self, video_info):
        video_info_map = {}
        for video_i in video_info:
            video_info_map[video_i['id']] = video_i
        return video_info_map
    
    def clear(self):
        # self.tracker = TrackBase(self.args)
        if len(self.results_track) > 0:
            res = []
            for video_name, cur_frame_id, boxes_xywh, score, ids, classes in self.results_track:
                res.append((cur_frame_id, boxes_xywh, ids, score, classes))
            result_filename = self.result_folder /  '{}.txt'.format(video_name)
            write_results(result_filename, res)
        self.results_track = []
    
    def update_record(self, target):
        cur_frame_id = target['frame_id'].item()
        cur_video_id = target['video_id'].item()
        # orig_size = target["orig_size"]
        if self.last_video_id == -1:
            self.last_video_id = cur_video_id
        if self.last_video_id == cur_video_id:
            assert cur_frame_id == self.last_frame_id + 1
            self.last_frame_id = cur_frame_id
        else:
            assert cur_frame_id == 1
            self.last_video_id = cur_video_id
            self.last_frame_id = cur_frame_id
            self.clear()

    
    def filter_dt_by_score(self, dt_instances: Instances) -> Instances:
        keep = dt_instances.scores > self.prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        keep &= dt_instances.labels > 0
        return dt_instances[keep]
    
    def filter_dt_by_area(self, dt_instances: Instances) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > self.area_threshold
        return dt_instances[keep]
    
    def result_post_process(self, targets, output):
        cur_frame_id = targets['frame_id'].item()
        cur_video_id = targets['video_id'].item()
        video_name = self.video_info_map[cur_video_id]['name']
        boxes = output[:, :4]
        score = output[:, 4]
        ids = output[:, 5]
        label = output[:, 6]

        if 'MOT17' in video_name or 'MOT20' in video_name:
            select_idx = label <= 1
            label = label[select_idx]
            boxes = boxes[select_idx]
            score = score[select_idx]
            ids = ids[select_idx]

        res_det = {}
        res_det['boxes'] = torch.tensor(boxes)
        res_det['scores'] = torch.tensor(score)
        res_det['labels'] = torch.tensor(label)

        boxes_xywh = deepcopy(boxes)
        boxes_xywh[:, 2:] = boxes_xywh[:, 2:] - boxes_xywh[:, :2]
        res_track = (video_name, cur_frame_id, boxes_xywh, score, ids, label)

        return res_det, res_track
    
    def prepare_for_coco_eval(self, dt_instances):
        num_select = min(len(dt_instances), self.num_select)
        scores = dt_instances.scores
        boxes = dt_instances.boxes
        labels = dt_instances.labels
        if self.args.amp:
            scores = scores.to('cuda')
        scores, topk_indexes = torch.topk(scores, num_select, dim=0)
        if self.args.amp:
            scores = scores.cpu()
            topk_indexes = topk_indexes.cpu()
        res_det = {}
        res_det['boxes'] = boxes[topk_indexes]
        res_det['scores'] = scores
        res_det['labels'] = labels[topk_indexes]
        return res_det
    
    def update(self, dt_instances_list, targets_list):
        det_result = []
        for dt_instances, targets in zip(dt_instances_list, targets_list):
            if self.num >= self.valid_frame_num:
                break
            self.num += 1
            det_result.append((
                targets['image_id'].item(), self.prepare_for_coco_eval(dt_instances)
                ))
                
            self.update_record(targets)
            dt_instances = self.filter_dt_by_score(dt_instances)
            dt_instances = self.filter_dt_by_area(dt_instances)

            num_occlusion = (dt_instances.labels == 0).sum()
            dt_instances.scores[dt_instances.labels == 0] *= -1

            self.total_dts += len(dt_instances)
            self.total_occlusion_dts += num_occlusion

            id = dt_instances.obj_idxes + 1
            tracker_outputs = torch.cat([dt_instances.boxes, dt_instances.scores[:, None], id[:, None], dt_instances.labels[:, None]], dim=-1).numpy()
            
            res_det, res_track = self.result_post_process(targets, tracker_outputs)
            # det_result.append((targets['image_id'].item(), res_det))
            self.results_track.append(res_track)
        if self.num >= self.valid_frame_num:
            self.clear()
        return det_result
