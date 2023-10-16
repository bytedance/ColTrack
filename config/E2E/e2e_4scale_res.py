_base_ = ['DINO_4scale_res.py']

USE_PARALLEL=False
NUM_PARALLEL_CORES=4

visual_copy=True
visual_flush_epoch = 2
visual_env_dir = 'vis_ignore'
visual_port = 32345
visual_host = "127.0.0.1"
visual_platform= "TensorBoard"

train_engine='MotSimpleTrainer'
evaluator='evaluate_e2e'
train_dataloader='mot_train_dataloader'
test_dataloader='mot_test_dataloader'

test_batch_size=4
save_checkpoint_interval=1
num_workers=8
batch_size=1
test_step=1
save_results = True
data_aug_max_size = 1440
backbone_dir='mot_files/models/dino'
pretrain_model_path = 'mot_files/models/dino/checkpoint0033_4scale.pth'

dataset_file="mot17_mix_ablation_mid"

multi_step_lr = True
lr_drop_list = [5, 10]
aug_step=25
resize_step=20
epochs=30


sampler_lengths=[2,3,3,3]
sample_mode='random_interval'
sample_interval=10
sampler_steps=[4, 8, 12]

prob_threshold = 0.6

# RuntimeTrackerBase
score_thresh=0.6
filter_score_thresh=0.6
miss_tolerance=5
