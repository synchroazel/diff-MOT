import argparse

from tqdm import tqdm

from motclass import MotDataset

dataset = "MOT20"
classification = True

# %% CLI args parser
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='python train.py',
    description='Script for training a graph network on the MOT task',
    epilog='Es: python train.py',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-D', '--datapath', default="/media/dmmp/vid+backup/Data",
                    help="Path to the folder containing the MOT datasets."
                         "NB: This project assumes a MOT dataset, this project has been tested with MOT17 and MOT20")
parser.add_argument('--model_savepath', default="saves/models",
                    help="Folder where models are loaded.")
parser.add_argument('--output_savepath', default="saves/outputs",
                    help="Folder where outputs are saved.")
parser.add_argument('--detection_gt_folder', default="gt",
                    help="Folder containing the ground truth detections files.")
parser.add_argument('--detection_gt_file', default="gt.txt",
                    help="Name of the ground truth detections file.")
parser.add_argument('--subtrack_len', default=20, type=int,
                    help="Length of the subtrack."
                         "NB: a value higher than 20 might require too much memory.")
parser.add_argument('--linkage_window', default=5, type=int,
                    help="Linkage window for building the graph."
                         "(e.s. if = 5 -> detections in frame 0 will connect to detections up to frame 5)")
parser.add_argument('--slide', default=15, type=int,
                    help="Sliding window to adopt during testing."
                         "NB: suggested to be subtrack_len - linkage_window")
parser.add_argument('-k', '--knn', default=20, type=int,
                    help="K parameter for knn reduction."
                         "NB: a value lower than 20 may exclude ground truths. Set to 0 for no knn.")
parser.add_argument('--cosine', action='store_true',
                    help="Use cosine distance instead of euclidean distance. Unavailable under MPS.")
parser.add_argument('--classification', action='store_true',
                    help="Work in classification setting instead of regression.")

args = parser.parse_args()

if args.knn <= 0:
    knn_args = None
else:
    knn_args = {
        'k': args.knn,
        'cosine': args.cosine
    }

data_loader = MotDataset(dataset_path=args.datapath,
                         split=args.split,
                         subtrack_len=args.subtrack_len,
                         slide=args.slide,
                         linkage_window=args.linkage_window,
                         detections_file_folder=args.gt_folder,
                         detections_file_name=args.gt_file,
                         dl_mode=args.dl_mode,
                         knn_pruning_args=knn_args,
                         classification=args.classification,
                         preprocessed=False,
                         preprocessing=True)

pbar_dl = tqdm(enumerate(data_loader), desc='[TQDM] Preprocessing subtracks ', total=data_loader.n_subtracks)

for _, _ in pbar_dl:
    cur_track_idx = data_loader.cur_track + 1
    cur_track_name = data_loader.tracklist[data_loader.cur_track]
    pbar_dl.set_description(
        f'[TQDM] Preprocessing subtrack {cur_track_idx}/{len(data_loader.tracklist)} ({cur_track_name})')
