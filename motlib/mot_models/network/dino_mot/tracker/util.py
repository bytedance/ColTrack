from motlib.mot_models.structures import Instances
from util import box_ops

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class RuntimeTrackerBase(object):
    def __init__(self, args):
        self.args = args
        self.score_thresh = self.args.score_thresh
        self.filter_score_thresh = self.args.filter_score_thresh
        self.miss_tolerance = self.args.miss_tolerance
        self.max_obj_id = 0

        self.last_frame_id = 0
        self.last_video_id = -1
    
    def update_record(self, target):
        cur_frame_id = target['frame_id'].item()
        cur_video_id = target['video_id'].item()
        # orig_size = target["orig_size"]
        if self.last_video_id == -1:
            self.last_video_id = cur_video_id
        if self.last_video_id == cur_video_id:
            assert cur_frame_id == self.last_frame_id + 1 or cur_frame_id == 1, 'cur frame id {}, last frame id {}, video id {}'.format(cur_frame_id, self.last_frame_id, cur_video_id)
            self.last_frame_id = cur_frame_id
        else:
            assert cur_frame_id == 1
            self.last_video_id = cur_video_id
            self.last_frame_id = cur_frame_id
            # self.clear()

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        # prob = out_logits[...,:1].sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels
        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances