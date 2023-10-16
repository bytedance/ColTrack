_base_ = ['base.py']

train_dataset = {'MOT17':['train_half'], 'Crowdhuman':['train', 'val']}

test_dataset = {'MOT17': ['val_half']}

num_classes = 91