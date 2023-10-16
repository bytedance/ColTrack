_base_ = ['DINO_5scale_res.py']

multi_step_lr = True
lr_drop_list = [20, 30]
aug_step=25
resize_step=20
resize_range=36
epochs=30

num_workers=8
save_checkpoint_interval=2
test_batch_size=4
test_step=2
finetune_ignore = []
data_aug_max_size = 1440
USE_PARALLEL=False
NUM_PARALLEL_CORES=4

det_thresh_offset=0.1
scores_low=0.2
match_thresh=0.9
match_thresh_second=0.7
track_nmsthre=0.8
track_confthre=0.2
track_buffer_offset=5
track_thresh_offset=-0.05
min_box_area = 100

visual_copy=True
visual_flush_epoch = 2
visual_env_dir = 'vis_ignore'
visual_port = 31094
visual_host = "127.0.0.1"
visual_platform= "TensorBoard"
save_results = True
backbone_dir='mot_files/models/dino'
pretrain_model_path='mot_files/models/dino/checkpoint0031_5scale.pth'

train_engine='SimpleTrainer'
evaluator='my_evaluate_coco'
train_dataloader='yolox_train_dataloader'
test_dataloader='default_test_dataloader'

dataset_file="mot17_mix_ablation_mid"