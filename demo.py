import argparse
from pathlib import Path
from motlib.utils import get_args_parser
from motlib.engine.task_engine import build_engin
import torch.distributed as dist
from motlib.utils import task_setup
from motlib.utils.video_tools.video2img import video2images_list
from motlib.utils.video_tools.dict2video import txt2img
from motlib.utils.video_tools.img2video import img2video


def main(args):
    task_setup(args)
    trainer = build_engin(args.train_engine, args)
    trainer.test()
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
    
    return trainer.data_loader_val.dataset.coco.dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ColTrack inference script', parents=[get_args_parser()])
    parser.add_argument('--infer_data_path', type=str, default=None)
    parser.add_argument('--is_mp4', action='store_true')
    parser.add_argument('--draw_tracking_results', action='store_true')
    parser.add_argument('--inference_sampler_interval', default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 25, 30, 36, 50, 60, 90], help='The downsampling interval of the videos.')
    args = parser.parse_args()
    args.eval = True
    args.save_log = True
    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.infer_data_path is None:
        print('Please use these --infer_data_path to specify the data path that needs to be processed')
        exit()

    if args.is_mp4:
        video_frames_path = output_dir / 'video_frames'
        
        if not video_frames_path.exists():
            video_frames_path.mkdir(exist_ok=True, parents=True)
            print(f'The decompressed video frames will be stored in {str(video_frames_path)}')
            video2images_list(args.infer_data_path, video_frames_path, num_threads=4)
        else:
            print(f'The decompressed video frames are already in {str(video_frames_path)}')
        args.data_root = str(video_frames_path)
    else:
        args.data_root = args.infer_data_path

    coco_dict = main(args)

    if args.draw_tracking_results:
        track_results_dir = output_dir / 'ColTrack/track_results'
        frames_draw_dir = output_dir / 'draw_track_results_frames'
        video_draw_dir = output_dir / 'draw_track_results_video'
        print(f'The tracking visualizations of each frame will be saved to frame: {str(frames_draw_dir)}. video: {video_draw_dir}')
        txt2img(track_results_dir, coco_dict, frames_draw_dir)
        img2video(frames_draw_dir, video_draw_dir)
