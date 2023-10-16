from .byte_tracker import BYTETracker
from collections import defaultdict
from motlib.utils import to_cpu
import torch
import torchvision
from pathlib import Path
from motlib.tracker import TRACKER_REGISTRY


def write_results(filename, results):
    filename = str(filename)
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{c},-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores, classes in results:
            for tlwh, track_id, score, class_id in zip(tlwhs, track_ids, scores, classes):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=int(frame_id), id=int(track_id), x1=round(x1, 3), y1=round(y1, 3), w=round(w, 3), h=round(h, 3), s=round(score, 5), c=int(class_id))
                f.write(line)
    print('save results to {}'.format(filename))


@TRACKER_REGISTRY.register()
class ByteTracker(object):
    def __init__(self, args, videos_info, result_folder) -> None:
        self.args = args
        self.num_classes = self.args.num_classes
        self.confthre = self.args.track_confthre
        self.nmsthre = self.args.track_nmsthre

        self.video_param = self.init_param_for_video(videos_info)
        self.video_name = defaultdict()
        self.results = []
        self.result_folder = Path(result_folder)
        self.result_folder.mkdir(parents=True, exist_ok=True)
        self.last_frame_id = 0
    
    def init_param_for_video(self, videos_info):
        new_videos_info = {}
        for video_info in videos_info:
            new_videos_info[video_info['id']] = video_info

        track_buffer_map = {
        }
        track_thresh_map = {
        }
        det_thresh_map = {
        }

        output = {}
        for k, v in new_videos_info.items():
            output[k] = {}
            output[k]['name'] = v['name']
            output[k]['frame_rate'] = v['frame_rate']
            output[k]['track_buffer'] = track_buffer_map.get(v['name'], 30)
            output[k]['track_thresh'] = track_thresh_map.get(v['name'], self.args.track_thresh_default)
            output[k]['det_thresh'] = det_thresh_map.get(v['name'], self.args.det_thresh_default)
        return output
            
    @torch.no_grad()
    def preprocess(self, target, result):
        image_id = int(to_cpu(target['image_id']))
        frame_id = int(to_cpu(target['frame_id']))
        video_id = int(to_cpu(target['video_id']))
        video_name = self.video_param[video_id]['name']
        self.args.track_buffer = self.video_param[video_id]['track_buffer']
        self.args.track_thresh = self.video_param[video_id]['track_thresh']
        self.args.frame_rate = self.video_param[video_id]['frame_rate']
        self.args.det_thresh = self.video_param[video_id]['det_thresh']
        self.args.match_thresh = self.args.match_thresh_default
        self.args.match_thresh_second = self.args.match_thresh_second_default

        boxes = result['boxes']
        
        if not boxes.size(0):
            return frame_id, video_id, video_name, None
        
        class_conf, class_pred = result['scores'], result['labels']
        class_conf = class_conf.view((-1, 1))
        class_pred = class_pred.view((-1, 1))
        conf_mask = (class_conf.squeeze() >= self.confthre).squeeze()
        label_mask = (class_pred.squeeze() > 0).squeeze()
        detections = torch.cat((boxes, class_conf, class_pred.float()), 1)
        detections = detections[conf_mask & label_mask]

        if not detections.size(0):
            return frame_id, video_id, video_name, None

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4],
            detections[:, 5],
            self.nmsthre,
        )
        detections = to_cpu(detections[nms_out_index])

        return frame_id, video_id, video_name, detections
    
    @torch.no_grad()
    def to_coco(self, detections, img_id):
        if detections is None:
            return None
        detections = detections
        data_list = []

        for ind in range(detections.shape[0]):
            pred_data = {
                "image_id": int(img_id),
                "category_id": int(detections[ind, 5]),
                "bbox": detections[ind, 0:4].tolist(),
                "score": detections[ind, 4].item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
        return data_list


    @torch.no_grad()
    def track(self, target, result, is_final):
        frame_id, video_id, video_name, detections = self.preprocess(target, result)

        if video_name not in self.video_name:
            self.video_name[video_id] = video_name
        if frame_id == 1:
            if 'MOT20' in video_name:
                self.args.mot20 = True
            else:
                self.args.mot20 = False
            
            self.tracker = BYTETracker(self.args)
            if len(self.results) != 0:
                result_filename = self.result_folder /  '{}.txt'.format(self.video_name[video_id - 1])
                write_results(result_filename, self.results)
                self.results = []
            self.last_frame_id = 0
        assert frame_id == self.last_frame_id + 1
        if detections is not None:
            online_targets = self.tracker.update(detections)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                class_id = t.class_id
                if 'MOT17' in video_name or 'MOT20' in video_name:
                    transverse = tlwh[2] / tlwh[3] > 1.6
                else:
                    transverse = False
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not transverse:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_cls.append(class_id)
                    online_scores.append(t.score)
            self.results.append((frame_id, online_tlwhs, online_ids, online_scores, online_cls))
        
        if is_final:
            result_filename = self.result_folder /  '{}.txt'.format(self.video_name[video_id])
            write_results(result_filename, self.results)
        self.last_frame_id = frame_id
        