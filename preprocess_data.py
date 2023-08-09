import argparse
import os
import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from motclass import MotDataset
from utilities import *

# todo: cli args

dataset = "MOT17"
classification=True

data_loader = MotDataset(dataset_path="/media/dmmp/vid+backup/Data/MOT17",
                         split="train",
                         subtrack_len=20,
                         slide=15,
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