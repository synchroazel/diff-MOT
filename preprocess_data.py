import argparse
import os
import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from motclass import MotDataset
from utilities import *

#'alexnet',
#  'convnext_base',
#  'convnext_large',
#  'convnext_small',
#  'convnext_tiny',
#  'deeplabv3_mobilenet_v3_large',
#  'deeplabv3_resnet101',
#  'deeplabv3_resnet50',
#  'densenet121',
#  'densenet161',
#  'densenet169',
#  'densenet201',
#  'efficientnet_b0',
#  'efficientnet_b1',
#  'efficientnet_b2',
#  'efficientnet_b3',
#  'efficientnet_b4',
#  'efficientnet_b5',
#  'efficientnet_b6',
#  'efficientnet_b7',
#  'efficientnet_v2_l',
#  'efficientnet_v2_m',
#  'efficientnet_v2_s',
#  'fasterrcnn_mobilenet_v3_large_320_fpn',
#  'fasterrcnn_mobilenet_v3_large_fpn',
#  'fasterrcnn_resnet50_fpn',
#  'fasterrcnn_resnet50_fpn_v2',
#  'fcn_resnet101',
#  'fcn_resnet50',
#  'fcos_resnet50_fpn',
#  'googlenet',
#  'inception_v3',
#  'keypointrcnn_resnet50_fpn',
#  'lraspp_mobilenet_v3_large',
#  'maskrcnn_resnet50_fpn',
#  'maskrcnn_resnet50_fpn_v2',
#  'maxvit_t',
#  'mc3_18',
#  'mnasnet0_5',
#  'mnasnet0_75',
#  'mnasnet1_0',
#  'mnasnet1_3',
#  'mobilenet_v2',
#  'mobilenet_v3_large',
#  'mobilenet_v3_small',
#  'mvit_v1_b',
#  'mvit_v2_s',
#  'quantized_googlenet',
#  'quantized_inception_v3',
#  'quantized_mobilenet_v2',
#  'quantized_mobilenet_v3_large',
#  'quantized_resnet18',
#  'quantized_resnet50',
#  'quantized_resnext101_32x8d',
#  'quantized_resnext101_64x4d',
#  'quantized_shufflenet_v2_x0_5',
#  'quantized_shufflenet_v2_x1_0',
#  'quantized_shufflenet_v2_x1_5',
#  'quantized_shufflenet_v2_x2_0',
#  'r2plus1d_18',
#  'r3d_18',
#  'raft_large',
#  'raft_small',
#  'regnet_x_16gf',
#  'regnet_x_1_6gf',
#  'regnet_x_32gf',
#  'regnet_x_3_2gf',
#  'regnet_x_400mf',
#  'regnet_x_800mf',
#  'regnet_x_8gf',
#  'regnet_y_128gf',
#  'regnet_y_16gf',
#  'regnet_y_1_6gf',
#  'regnet_y_32gf',
#  'regnet_y_3_2gf',
#  'regnet_y_400mf',
#  'regnet_y_800mf',
#  'regnet_y_8gf',
#  'resnet101',
#  'resnet152',
#  'resnet18',
#  'resnet34',
#  'resnet50',
#  'resnext101_32x8d',
#  'resnext101_64x4d',
#  'resnext50_32x4d',
#  'retinanet_resnet50_fpn',
#  'retinanet_resnet50_fpn_v2',
#  's3d',
#  'shufflenet_v2_x0_5',
#  'shufflenet_v2_x1_0',
#  'shufflenet_v2_x1_5',
#  'shufflenet_v2_x2_0',
#  'squeezenet1_0',
#  'squeezenet1_1',
#  'ssd300_vgg16',
#  'ssdlite320_mobilenet_v3_large',
#  'swin3d_b',
#  'swin3d_s',
#  'swin3d_t',
#  'swin_b',
#  'swin_s',
#  'swin_t',
#  'swin_v2_b',
#  'swin_v2_s',
#  'swin_v2_t',
#  'vgg11',
#  'vgg11_bn',
#  'vgg13',
#  'vgg13_bn',
#  'vgg16',
#  'vgg16_bn',
#  'vgg19',
#  'vgg19_bn',
#  'vit_b_16',
#  'vit_b_32',
#  'vit_h_14',
#  'vit_l_16',
#  'vit_l_32',
#  'wide_resnet101_2',
#  'wide_resnet50_2'







# todo: cli args
parser = argparse.ArgumentParser(
    prog='python train.py',
    description='Script for training a graph network on the MOT task',
    epilog='Es: python train.py',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-m', '--mot', default="MOT17", type=str)
parser.add_argument('--classification', action='store_true')
parser.add_argument('--backbone',  default="resnet50", type=str)

args = parser.parse_args()



dataset = args.mot
classification = args.classification
backbone = args.backbone

data_loader = MotDataset(dataset_path="/media/dmmp/vid+backup/Data/" + dataset,
                         split="train",
                         subtrack_len=15,
                         slide=10,
                         linkage_window=5,
                         detections_file_folder="gt",
                         detections_file_name="gt.txt",
                         dl_mode=True,
                         knn_pruning_args={"k": 99, "cosine": False},
                         preprocessing=True,
                         preprocessed=False,
                         classification=classification,
                         feature_extraction_backbone=backbone)

pbar_dl = tqdm(enumerate(data_loader), desc='[TQDM] Preprocessing subtracks ', total=data_loader.n_subtracks)

for _, _ in pbar_dl:
    cur_track_idx = data_loader.cur_track + 1
    cur_track_name = data_loader.tracklist[data_loader.cur_track]
    pbar_dl.set_description(
        f'[TQDM] Preprocessing subtrack {cur_track_idx}/{len(data_loader.tracklist)} ({cur_track_name})')