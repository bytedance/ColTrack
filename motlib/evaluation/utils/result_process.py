# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import numpy as np
import torch


def filter_result(det_result, res, tracker_name, dataset_class_name, res_record):
    res = res[dataset_class_name][tracker_name]['COMBINED_SEQ']
    if 'cls_comb_cls_av' in res:
        res = res['cls_comb_cls_av']
    else:
        res = res['pedestrian']
    hota = res['HOTA']['HOTA']
    # hota = "{:.3f}".format(100 * np.mean(hota))
    hota = round(100 * np.mean(hota), 3)
    mota = round(100 * res['CLEAR']['MOTA'], 3)
    ids = res['CLEAR']['IDSW']
    idf1 = round(100 * res['Identity']['IDF1'], 3)
    
    if det_result is not None:
        res_record[0] = round(det_result['coco_eval_bbox'][0] * 100, 3)
    else:
        res_record[0] = -1
    res_record[1] = hota
    res_record[2] = mota
    res_record[3] = ids
    res_record[4] = idf1
    return res_record
