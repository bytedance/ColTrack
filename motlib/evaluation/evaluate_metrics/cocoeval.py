# Copyright (2023) Bytedance Ltd. and/or its affiliates 



from datasets.coco_eval import CocoEvaluator as DefaultCocoEvaluator
import logging
import numpy as np


class CocoEvaluator(DefaultCocoEvaluator):
    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            if len(self.eval_imgs[iou_type]) == 0:
                self.eval_imgs[iou_type] = [np.zeros((1, 4, 0))]
        super().synchronize_between_processes()

    def summarize(self):
        super().summarize()
        stats = self.coco_eval['bbox'].stats * 100.0
        logger = logging.getLogger(__name__)
        logger.info("\n\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n".format(*(stats.tolist())))

