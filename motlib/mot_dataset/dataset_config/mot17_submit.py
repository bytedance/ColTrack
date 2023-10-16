_base_ = ['base.py']

train_dataset = {'MOT17':['train'], 'Crowdhuman':['train', 'val']}

test_dataset = {'MOT17':['test']}

num_classes = 91