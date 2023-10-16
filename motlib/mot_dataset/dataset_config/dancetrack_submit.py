_base_ = ['base.py']

train_dataset = {'DanceTrack':['train', 'val'], 'Crowdhuman':['train', 'val']}

test_dataset = {'DanceTrack': ['test']}

num_classes = 91