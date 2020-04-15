from os import path

data_path = '/data/leavesboxes1image/'

annotatio_path = path.join(data_path,'via_export_coco_cat.json')

image_path = path.join(data_path,'20200307_160518.jpg')

num_epochs = 10

num_classes = 2

train_batch_size = 1
train_shuffle_dl = True
num_workers_dl = 4


use_cuda=True

lr= 0.01
momentum= 0.9
weight_decay= 0.005
