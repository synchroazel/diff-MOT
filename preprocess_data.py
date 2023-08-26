import argparse
import os

from tqdm import tqdm

from motclass import MotDataset

parser = argparse.ArgumentParser()

parser.add_argument('-D', '--datapath', default="data", type=str)  # TODO: remove default
parser.add_argument('-m', '--mot', default="MOT17", type=str)
parser.add_argument('--regression', action='store_true')
parser.add_argument('--detector', action='store_true')
parser.add_argument('--submission', action='store_true')
parser.add_argument('--backbone', default="efficientnet_v2_l", type=str)
parser.add_argument('--split', default='train', type=str)
parser.add_argument('--detections', default="gt", type=str)

args = parser.parse_args()

assert args.split in ['train', 'test']
assert args.detections in ['gt', 'det']

datapath = args.datapath
dataset = args.mot
classification = not args.regression
backbone = args.backbone
split = args.split
det_folder = args.detections
det_file = args.detections + ".txt"

dataloader = MotDataset(dataset_path=os.join(datapath, dataset),
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

pbar_dl = tqdm(enumerate(dataloader), desc='[TQDM] Preprocessing subtracks ', total=dataloader.n_subtracks)

for _, _ in pbar_dl:
    cur_track_idx = dataloader.cur_track + 1
    cur_track_name = dataloader.tracklist[dataloader.cur_track]
    pbar_dl.set_description(
        f'[TQDM] Preprocessing subtrack {cur_track_idx}/{len(dataloader.tracklist)} ({cur_track_name})')
