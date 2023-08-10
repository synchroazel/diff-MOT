import argparse
import os
import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from motclass import MotDataset
from utilities import *

# todo: cli args
parser = argparse.ArgumentParser(
    prog='python train.py',
    description='Script for training a graph network on the MOT task',
    epilog='Es: python train.py',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-m', '--mot', default="MOT17", type=str)
parser.add_argument('--classification', action='store_true')

args = parser.parse_args()



dataset = args.mot
classification = args.classification

data_loader = MotDataset(dataset_path="/media/dmmp/vid+backup/Data/" + dataset,
                         split="train",
                         subtrack_len=15,
                         slide=10,
                         linkage_window=5,
                         detections_file_folder="gt",
                         detections_file_name="gt.txt",
                         dl_mode=True,
                         knn_pruning_args={"k": 20, "cosine": False},
                         preprocessing=True,
                         preprocessed=False,
                         classification=classification)

pbar_dl = tqdm(enumerate(data_loader), desc='[TQDM] Preprocessing subtracks ', total=data_loader.n_subtracks)

for _, _ in pbar_dl:
    cur_track_idx = data_loader.cur_track + 1
    cur_track_name = data_loader.tracklist[data_loader.cur_track]
    pbar_dl.set_description(
        f'[TQDM] Preprocessing subtrack {cur_track_idx}/{len(data_loader.tracklist)} ({cur_track_name})')