_base_ = ['DINO_mot_5scale.py']

data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 20000

track_thresh_default = 0.6
det_thresh_default = 0.65
scores_low=0.25
match_thresh_default=0.8
match_thresh_second_default=0.6
track_nmsthre=0.9
track_confthre=0.2