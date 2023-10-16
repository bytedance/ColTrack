_base_ = ['base.py']

train_dataset = {'DanceTrack':['train'], 'Crowdhuman':['train', 'val']}

test_dataset = {'DanceTrack': ['val']}

num_classes = 91