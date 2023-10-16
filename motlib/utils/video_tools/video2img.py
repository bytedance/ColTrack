# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import cv2
import os
import threading
from pathlib import Path


def print_video_info(video):
    # frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    # frames
    fps = int(round(video.get(cv2.CAP_PROP_FPS)))
    vw = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vn = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Raw FPS: "+str(fps))
    print("Raw number of frames: "+str(vn))
    print("Raw Total time: "+"{0:.2f}".format(vn/fps)+"s")
    print("Rae size wh:[{}, {}]".format(vw, vh))


def video2images(video_path, images_folder, scale=None, frame_frequency=None, thread_flag=False):
    assert os.path.exists(video_path) and os.path.isfile(video_path)
    img_id = 0
    
    # 提取视频的频率
    
	# 如果文件目录不存在则创建目录
    os.makedirs(images_folder, exist_ok=True)
        
    # 读取视频帧
    camera = cv2.VideoCapture(video_path)
    fps = camera.get(cv2.CAP_PROP_FPS)
    if frame_frequency is not None:
        sample_frequency = int(fps) // frame_frequency
    else:
        sample_frequency = 1
    if not thread_flag:
        print_video_info(camera)
    print_frequency = 50
    while True:
        img_id += 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        if scale is not None:
            image = cv2.resize(image, None, fx=scale, fy=scale)
        if img_id % sample_frequency == 0:
            cv2.imwrite(images_folder + '/' + "{:0>6d}.jpg".format(img_id), image)
        if img_id % print_frequency == 0 and not thread_flag:
            print('frame extracting id {}'.format(img_id))
            
    camera.release()
    print(f'frames extracting finished from {video_path} ----> {images_folder}')


def video2images_thread(video_path_list, images_folders, scale, frame_frequencye, thread_flag):
    for video_path, img_folder in zip(video_path_list, images_folders):
        video2images(video_path=str(video_path), images_folder=str(img_folder), scale=scale, frame_frequency=frame_frequencye,thread_flag=thread_flag)

def filter_folder(file_list, name_list):
    res = []
    for f in file_list:
        drop_flag = False
        for n in name_list:
            if n in str(f):
                drop_flag = True
                continue
        if not drop_flag:
            res.append(f)
    return res

def video2images_list(video_list_or_folder, output_folder, scale=None, frame_frequency=None, num_threads = 1, name_filter=[]):
    if isinstance(video_list_or_folder, (list, tuple)):
        video_list = video_list_or_folder
        video_list = filter_folder(video_list, name_filter)
        img_folders = [output_folder for _ in video_list]
    else:
        video_list_or_folder = Path(video_list_or_folder).resolve()
        video_list = video_list_or_folder.rglob('*.*')
        video_list = [video_dir.resolve() for video_dir in video_list if video_dir.suffix.lower() in ['.mp4']]
        video_list = filter_folder(video_list, name_filter)

        output_folder = Path(output_folder)
        img_folders = []
        for video_dir in video_list:
            img_folder = output_folder / str(video_dir.parent)[len(str(video_list_or_folder))+1:] / video_dir.stem
            img_folders.append(img_folder)

    thread_flag = True if num_threads > 1 else False
    
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=video2images_thread, args=(
            video_list[i::num_threads], 
            img_folders[i::num_threads],
            scale, frame_frequency, thread_flag))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Total {} vide has been processed!".format(len(video_list)))



if __name__ == '__main__':
    input_dirs = ''
    output_dir = ''
    
    video2images_list(input_dirs, output_dir, num_threads=32)