import os

import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss
from model import Net
from motclass import MotDataset
from utilities import get_best_device, save_model
from torch.nn import BCEWithLogitsLoss, BCELoss

device = get_best_device()


# %% Function definitions

def single_validate(model, val_loader, idx, loss_function, device):
    """ Validate the model on a single subtrack, given a MOT dl and an index"""
    model.eval()

    data = val_loader[idx]

    with torch.no_grad():
        data = ToDevice(device.type)(data)
        pred_edges = model(data)  # Get the predicted edge labels
        gt_edges = data.y  # Get the true edge labels

        # Check how many gt 1s have been predicted correctly
        gt_ones = gt_edges.nonzero()
        acc_on_ones = torch.where(pred_edges[gt_ones] > 0.5, 1., 0.).mean()

        # Check how many gt 0s have been predicted correctly
        gt_zeros = (gt_edges == 0).nonzero()
        acc_on_zeros = torch.where(pred_edges[gt_zeros] < 0.5, 1., 0.).mean()

        # Check how many gt values overall have been predicted correctly
        rounded_pred_edges = torch.where(pred_edges > 0.5, 1., 0.)
        acc_total = torch.where(rounded_pred_edges == gt_edges, 1., 0.).mean()

        loss = loss_function(pred_edges, gt_edges, reduction='sum', alpha=0.9, gamma=5)

        return loss.item(), acc_total, acc_on_ones, acc_on_zeros


def train(model, train_loader, val_loader, loss_function, optimizer, epochs, device, mps_fallback=False):
    model = model.to(device)
    model.train()

    print('[INFO] Launching training...\n')

    pbar_ep = tqdm(range(epochs), desc='[TQDM] Epoch #1 ', position=0, leave=False,
                   bar_format="{desc:<5}{percentage:3.0f}%|{bar}{r_bar}")

    for epoch in pbar_ep:

        total_train_loss, total_val_loss = 0, 0

        pbar_dl = tqdm(enumerate(train_loader), desc='[TQDM] Training on track 1/? ', total=train_loader.n_subtracks,
                       bar_format="{desc:<5}{percentage:3.0f}%|{bar}{r_bar}")

        last_track_idx = 0
        i = 0
        for i, data in pbar_dl:
            data = ToDevice(device.type)(data)

            cur_track_idx = train_loader.cur_track + 1
            cur_track_name = train_loader.tracklist[train_loader.cur_track]

            pbar_dl.set_description(
                f'[TQDM] Training on track {cur_track_idx}/{len(train_loader.tracklist)} ({cur_track_name})')

            # On track switch, save the model
            if cur_track_idx != last_track_idx:
                save_model(model, mps_fallback=mps_fallback)

            """ Training step """

            pred_edges = model(data)  # Get the predicted edge labels
            gt_edges = data.y  # Get the true edge labels

            train_loss = loss_function(pred_edges, gt_edges, reduction='sum', alpha=0.9, gamma=5)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            total_train_loss += train_loss.item()

            pbar_dl.update(1)

            """ Validation step """

            val_loss, val_acc, val_acc_1, val_acc_0 = single_validate(model, val_loader, i, loss_function, device)
            total_val_loss += val_loss

            """ Update progress """

            avg_train_loss_msg = f'avg.Tr.Loss: {(total_train_loss / (i + 1)):.4f} (last: {train_loss:.4f})'
            avg_val_loss_msg = f'avg.Val.Loss: {(total_val_loss / (i + 1)):.4f} (last: {val_loss:.4f})'
            last_val_acc = f'Val.Acc: {val_acc:.4f} (1s: {int(val_acc_1 * 100)}% | 0s: {int(val_acc_0 * 100)}%)'

            pbar_ep.set_description(
                f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg} - {avg_val_loss_msg} | {last_val_acc}')

            last_track_idx = cur_track_idx

        pbar_ep.set_description(f'[TQDM] Epoch #{epoch + 1} - avg.Loss: {(total_train_loss / (i + 1)):.4f}')


# %% Set up parameters

# Paths
mot_path = 'data'
# mot_path = '/media/dmmp/vid+backup/Data'

# MOT to use
mot_train = 'MOT17'
mot_val = 'MOT20'

# Dtype to use
dtype = torch.float32

# Hyperparameters
backbone = 'resnet50'
layer_type = 'GeneralConv'
subtrack_len = 15
slide = 15
linkage_window = -1
l_size = 4098
epochs = 1
heads = 1
learning_rate = 0.0001

# Only if using MPS
mps_fallback = True
# mps_fallback = False

# %% Initialize the model

model = Net(backbone=backbone,
            layer_tipe=layer_type,
            layer_size=l_size,
            dtype=dtype,
            mps_fallback=True if device == torch.device('mps') else False,
            in_edge_channels=2,
            heads=heads,
            concat=False,
            dropout=0.3,
            add_self_loops=False,
            steps=6,
            device=device
            )

# loss_function = torch.nn.BCEWithLogitsLoss()
loss_function = sigmoid_focal_loss

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %% Set up the dataloader

train_dataset_path = os.path.normpath(os.path.join(mot_path, mot_train))
val_dataset_path = os.path.normpath(os.path.join(mot_path, mot_val))

mot_train_dl = MotDataset(dataset_path=train_dataset_path,
                          split='train',
                          subtrack_len=subtrack_len,
                          slide=slide,
                          linkage_window=linkage_window,
                          detections_file_folder='gt',
                          detections_file_name='gt.txt',
                          dl_mode=True,
                          device=device,
                          dtype=dtype,
                          mps_fallback=mps_fallback)

mot_val_dl = MotDataset(dataset_path=val_dataset_path,
                        split='train',
                        subtrack_len=subtrack_len,
                        slide=slide,
                        linkage_window=linkage_window,
                        detections_file_folder='gt',
                        detections_file_name='gt.txt',
                        dl_mode=True,
                        device=device,
                        dtype=dtype,
                        mps_fallback=mps_fallback)

# %% Train the model

train(model, mot_train_dl, mot_val_dl, loss_function, optimizer, epochs, device, mps_fallback)
