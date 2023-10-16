from pathlib import Path
from pycocotools.coco import COCO
from collections import defaultdict
from copy import deepcopy
import time
import json

class MotCOCO(COCO):
    def __init__(self, annotation_file = None):
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            if isinstance(annotation_file, dict):
                dataset = annotation_file
            else:
                with open(annotation_file, 'r') as f:
                    dataset = json.load(f)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        super().createIndex()
        self.eval_config = self.dataset['eval_config']

        videoToImgs = defaultdict(list)

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                videoToImgs[img['video_id']].append(img)
        
        videos = {}
        if 'videos' in self.dataset:
            for video in self.dataset['videos']:
                video = deepcopy(video)
                if 'number' not in video:
                    video['number'] = len(videoToImgs[video['id']])
                videos[video['id']] = video

        
        self.videos = videos
        self.videoToImgs = videoToImgs

        videoFrameToImg = {}
        for video_id, imgs_info in self.videoToImgs.items():
            video_name = self.videos[video_id]['name']
            videoFrameToImg[video_name] = {}

            for img_info in imgs_info:
                frame_id = img_info['frame_id']
                videoFrameToImg[video_name][int(frame_id)] = img_info


        self.videoFrameToImg = videoFrameToImg
        