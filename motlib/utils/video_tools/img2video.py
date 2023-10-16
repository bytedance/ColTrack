import cv2
import os
import threading
import glob as gb
from pathlib import Path
from PIL import Image

def img2video(visual_path="", output_path='', size=None):
    print("Starting img2video")
    visual_path = Path(visual_path)
    output_dir = Path(output_path) / (visual_path.name + "_video.avi")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        output_dir.unlink()
    
    img_paths = visual_path.glob("*.*")
    img_paths = [f for f in img_paths if f.suffix in ['.jpg', '.png']]
    img_paths = sorted(img_paths)
    fps = 16 
    if size is None:
        if len(img_paths) > 0:
            img = Image.open(str(img_paths[0]))
            w, h = img.size
            size = (w, h)
        else:
            size = (1920,1080) 
    videowriter = cv2.VideoWriter(str(output_dir),cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

    for idx, img_path in enumerate(img_paths):
        if idx % 50 == 0 and idx != 0:
            print('Processing {} images'.format(idx))
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print(f"img2video Done. The video is stored in {str(output_path)}.")

def video2images_multi(video_folder, output_folder):
    count = 0
    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)
        threading.Thread(target=img2video, args=(video_path, output_folder)).start()
        count = count + 1
        print("{} th video {} has been finished!".format(count, video_name))

def video2images_list(video_list, output_folder):
    count = 0
    for video_path in video_list:
        threading.Thread(target=img2video, args=(video_path, output_folder)).start()
        count = count + 1
        print("{} th video {} has been finished!".format(count, video_path))


if __name__ == '__main__':
    img2video('', '')