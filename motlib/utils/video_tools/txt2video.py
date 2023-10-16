import os
import sys
import json
import cv2
import glob as gb
import numpy as np
from pathlib import Path
from collections import defaultdict


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def txt2img(video_path, visual_path, valid_labels={-1}, ignore_labels={}):
    print("Starting txt2img")
    color_list = colormap()

    video_dirs = list(Path(video_path).glob('*'))
    video_dirs.sort()

    for video_dir in video_dirs:
        visual_dir = Path(visual_path) / video_dir.name
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)

        txt_path = video_dir / 'gt' / 'gt.txt'

        txt_dict = defaultdict(list)   
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')

                pid = int(float(linelist[7]))
                tid = int(linelist[1])
                img_id = int(linelist[0])
                
                if pid in valid_labels or pid in ignore_labels:
                    continue

                bbox = [float(linelist[2]), float(linelist[3]), 
                        float(linelist[2]) + float(linelist[4]), 
                        float(linelist[3]) + float(linelist[5]), tid]
                txt_dict[int(img_id)].append(bbox)

        for img_id in sorted(txt_dict.keys()):
            img_dir = str(video_dir / 'img1' / "{:0>6d}.jpg".format(img_id))
            img = cv2.imread(img_dir)
            for bbox in txt_dict[img_id]:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[bbox[4]%79].tolist(), thickness=2)
                txt_info = str(int(bbox[4])) + '_' + str(pid)
                cv2.putText(img, txt_info, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_list[bbox[4]%79].tolist(), 2)
            img_out_dir = str(visual_dir / "{:0>6d}.png".format(img_id))
            cv2.imwrite(img_out_dir, img)
        print(video_dir, "Done")
        break
    print("txt2img Done")

        
def img2video(video_path, visual_path):
    print("Starting img2video")

    video_dirs = list(Path(video_path).glob('*'))
    video_dirs.sort()
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    for video_dir in video_dirs:
        visual_dir = Path(visual_path) / (video_dir.name + "_video.avi")

        img_paths = video_dir.glob("*.png")
        img_paths = [str(p) for p in list(img_paths)][:5000]
        fps = 16 
        size = (1920,1080) 
        videowriter = cv2.VideoWriter(str(visual_dir),cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

        for img_path in sorted(img_paths):
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, size)
            videowriter.write(img)

        videowriter.release()
    print("img2video Done")


if __name__ == '__main__':
    video_path = ""
    visual_path=""
    video_visual_path = ""
    if len(sys.argv) > 1:
        visual_path =sys.argv[1]
    txt2img(video_path, visual_path)
    img2video(visual_path,video_visual_path)
