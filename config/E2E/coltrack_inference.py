_base_ = ['e2e_4scale_res.py']

param_dict_type = 'finetune'
frozen_weights_mot = ['dino.backbone.0','dino.input_proj','dino.transformer.level_embed', 'dino.transformer.encoder']

finetune_ignore = ['transformer.decoder', 'tgt_embed.weight']

pretrain_model_path = None

lr_drop_list = [50, 200]
aug_step=2000
epochs=60

num_classes = 91 # 91 for mot17 and dancetrack, 9 for bdd100k
just_inference_flag = True
dataset_file = 'empty'
# inference_sampler_interval = 10  # select from [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 25, 30, 36, 50, 60, 90]

num_queries = 300
dec_n_points = 8
add_kalmanfilter = False
p_era = 0.0
area_keep = 0.5

track_instance_drop = 0.1
track_instance_fp = 0.3

fp_history_ratio = 0.0
inter_action_layer=0

transformer_name='TimeTrackDeformableTransformer'
qim_name = 'TimeTrackQIM'
track_criterion_name = 'TimeTrackMotSetCriterion'
mot_model_name = 'TimeTrackDINO'

sampler_lengths=[4, 4]
sampler_interval_scale = [1.0, 1.0]
sampler_steps=[0, 4]

sample_mode='random_interval'
sample_interval=10

mem_bank_len=3

prob_threshold = 0.6
# RuntimeTrackerBase
score_thresh=0.6
filter_score_thresh=0.6
miss_tolerance=22
test_batch_size=8

data_aug_scales = [800]
data_aug_max_size = 200000
