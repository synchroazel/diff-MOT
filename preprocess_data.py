import argparse
import os
import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from motclass import MotDataset
from utilities import *

#  'efficientnet_v2_l',

#  'resnet101',

#  'resnet50',

#  'vgg16',


#  'vit_l_32',




# todo: cli args
parser = argparse.ArgumentParser(
    prog='python train.py',
    description='Script for training a graph network on the MOT task',
    epilog='Es: python train.py',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-m', '--mot', default="MOT17", type=str)
parser.add_argument('--regression', action='store_true')
parser.add_argument('--detector', action='store_true')
parser.add_argument('--submission', action='store_true')
parser.add_argument('--backbone',  default="efficientnet_v2_l", type=str)

args = parser.parse_args()

# TODO: remove
#
args.submission = True
#


split = 'test' if args.submission else "train"
det_folder = 'det' if (args.submission or args.detector )else "gt"
det_file = 'det.txt' if (args.submission or args.detector ) else "gt.txt"


dataset = args.mot
classification = not args.regression
backbone = args.backbone

data_loader = MotDataset(dataset_path="/media/dmmp/vid+backup/Data/" + dataset,
                         split=split,
                         subtrack_len=15,
                         slide=10,
                         linkage_window=5,
                         detections_file_folder=det_folder,
                         detections_file_name=det_file,
                         dl_mode=True,
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