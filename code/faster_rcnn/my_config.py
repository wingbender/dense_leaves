from os import path

data_path = '/data/leavesboxes1image/'

annotatio_path = path.join(data_path,'via_export_coco_cat.json')

image_path = path.join(data_path,'20200307_160518.jpg')

crop_size = 300
stride = 200
split_direction = 'h'
splits = [.25,.20]  # [test,validation]

num_epochs = 3
print_freq = 4

num_classes = 2

train_batch_size = 3
train_shuffle_dl = True
num_workers_dl = 1  # There is an issue with more than one worker when loading images ¯\_(ツ)_/¯


scheduler_step = 4
scheduler_gamma = 0.1

use_cuda=True

lr= 0.001
momentum= 0.9
weight_decay= 0.005
