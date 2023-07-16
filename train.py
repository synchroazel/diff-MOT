import sys

import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from model import Net
from motclass import MotDataset
from utilities import get_best_device

device = get_best_device()


def validate(model, val_loader, idx, loss_function, device):
    """ Validate the model on a single subtrack, given a MOT dl and an index"""
    model.eval()

    data = val_loader[idx]

    with torch.no_grad():
        data = ToDevice(device.type)(data)
        pred_edges = model(data)  # Get the predicted edge labels
        gt_edges = data.y  # Get the true edge labels
        loss = loss_function(pred_edges, gt_edges)
        return loss.item()


def train(model, train_loader, loss_function, optimizer, epochs, device):
    model = model.to(device)
    model.train()

    print('[INFO] Launching training...\n')

    pbar_ep = tqdm(range(epochs), desc='[TQDM] Epoch #1 ', position=0, leave=False)

    for epoch in pbar_ep:

        total_train_loss, total_val_loss = 0, 0

        pbar_dl = tqdm(enumerate(train_loader), desc='[TQDM] Training on track 1/? ', total=train_loader.n_subtracks)

        for i, data in pbar_dl:
            data = ToDevice(device.type)(data)

            cur_track_idx = train_loader.cur_track + 1
            cur_track_name = train_loader.tracklist[train_loader.cur_track]

            pbar_dl.set_description(
                f'[INFO] Training on track {cur_track_idx}/{len(train_loader.tracklist)} ({cur_track_name})')

            """ Training step """

            pred_edges = model(data)  # Get the predicted edge labels
            gt_edges = data.y  # Get the true edge labels

            train_loss = loss_function(pred_edges, gt_edges)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            total_train_loss += train_loss.item()

            pbar_dl.update(1)

            """ Validation step """

            val_loader = train_loader
            val_loss = validate(model, val_loader, i + 1, loss_function, device)
            total_val_loss += val_loss

            """ Update progress """

            avg_train_loss_msg = f'avg.Tr.Loss: {(total_train_loss / (i + 1)):.4f} (last: {train_loss.item():.4f})'
            avg_val_loss_msg = f'avg.Val.Loss: {(total_val_loss / (i + 1)):.4f} (last: {val_loss:.4f})'

            pbar_ep.set_description(
                f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg} - {avg_val_loss_msg}')

        pbar_ep.set_description(f'[TQDM] Epoch #{epoch + 1} - avg.Loss: {(total_train_loss / (i + 1)):.4f}')


mot20_path = "data/MOT20"

# Hyperparameters
dtype = torch.float32
backbone = 'resnet50'
layer_type = 'GATConv'
subtrack_len = 15
slide = 5
linkage_window = subtrack_len//3
l_size = 128
epochs = 10
learning_rate = 0.001

model = Net(backbone=backbone,
            layer_tipe=layer_type,
            layer_size=l_size,
            dtype=dtype,
            mps_fallback=True if device == torch.device('mps') else False)

loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)

mo20_train_dl = MotDataset(dataset_path=mot20_path,
                           split='train',
                           subtrack_len=subtrack_len,
                           slide=slide,
                           linkage_window=5,
                           detections_file_folder='gt',
                           detections_file_name='gt.txt',
                           device=device,
                           dl_mode=True,
                           dtype=dtype)

train(model, mo20_train_dl, loss_function, optimizer, epochs, device)
