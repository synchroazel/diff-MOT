import os

import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from motclass_test import MotDataset
from utilities import get_best_device, load_model_pkl

device = get_best_device()


# %% Function definitions


def test(model, val_loader, loss_function, output_file_folder="outcomes", device="cuda"):

    def trajectories_to_csv():
        pass

    outcomes = dict()

    model = model.to(device)
    model.eval()

    print('[INFO] Launching test...\n')

    val_loss, total_val_loss = 0, 0

    pbar_dl = tqdm(enumerate(val_loader), desc='[TQDM] Testing on track 1/? ', total=val_loader.n_subtracks)

    for i, data in pbar_dl:
        cur_track_idx = val_loader.cur_track + 1
        cur_track_name = val_loader.tracklist[val_loader.cur_track]

        with torch.no_grad():
            data = ToDevice(device.type)(data)

            pred_edges = model(data)  # Get the predicted edge labels
            gt_edges = data.y  # Get the true edge labels
            val_loss = loss_function(pred_edges, gt_edges)

            zero_mask = pred_edges < 0.33
            one_mask = pred_edges > 0.5

            pred_edges = torch.where(one_mask , 1., pred_edges)
            pred_edges = torch.where(zero_mask, 0, pred_edges)

            zero_mask = gt_edges < 0.33
            one_mask = gt_edges > 0.5

            gt_edges = torch.where(one_mask, 1., gt_edges)
            gt_edges = torch.where(zero_mask, 0, gt_edges)

            acc_on_ones = torch.where(pred_edges[one_mask] == 1.0, 1., 0.).mean()
            acc_zeros = torch.where(pred_edges[zero_mask] == 0., 1., 0.).mean()
            zeros_as_ones = torch.where(pred_edges[one_mask] == 0., 1., 0.).mean() * 100
            ones_as_zeros = torch.where(pred_edges[zero_mask] == 1., 1., 0.).mean() * 100
            total_val_loss += val_loss

        avg_val_loss_msg = f'avg.Loss: {(total_val_loss / (i + 1)):.4f} (last: {val_loss:.4f})'

        last_val_acc = f'Last Val.Acc: {acc_on_ones:.4f} on ones, {acc_zeros:.4f} on zeroes, zeros as ones: {zeros_as_ones:.2f}%, one as zeros: {ones_as_zeros:.2f}%'

        pbar_dl.set_description(
            f'[TQDM] Testing on track {cur_track_idx}/{len(val_loader.tracklist)} ({cur_track_name}) - {avg_val_loss_msg} | {last_val_acc}')


# %% Set up parameters

# Paths
# mot_path = 'data'
mot_path = '/media/dmmp/vid+backup/Data'
saves_path = 'saves/models'

# Model to load
model_pkl = 'timeaware_500_resnet50-backbone.pkl'

# MOT to use
mot = 'MOT20'

# Dtype to use
dtype = torch.float32

# Hyperparameters
backbone = 'resnet50'
layer_type = 'TransformerConv'
subtrack_len = 20
slide = 15
linkage_window = 5
l_size = 128
epochs = 1
learning_rate = 0.001

# Only if using MPS
mps_fallback = False


# %% Load the model

model_path = os.path.normpath(os.path.join(saves_path, model_pkl))

model = load_model_pkl(model_path, device='cuda').to(device)

model.mps_fallback = mps_fallback

# loss_function = torch.nn.BCEWithLogitsLoss()
loss_function = torch.nn.HuberLoss(reduction="mean", delta=.4)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)

# %% Set up the dataloader

dataset_path = os.path.normpath(os.path.join(mot_path, mot))

mot_train_dl = MotDataset(dataset_path=dataset_path,
                          split='train',
                          subtrack_len=subtrack_len,
                          slide=slide,
                          linkage_window=linkage_window,
                          detections_file_folder='gt',
                          detections_file_name='gt.txt',
                          dl_mode=True,
                          device=device,
                          mps_fallback=mps_fallback,
                          dtype=dtype)

# %% Test the model

test(model=model, val_loader=mot_train_dl, loss_function=loss_function, device=device)
