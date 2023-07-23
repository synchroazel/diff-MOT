import os

import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss
from model import Net
from motclass import MotDataset
from utilities import get_best_device, save_model
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
        loss = loss_function(pred_edges, gt_edges, reduction='mean')
        return loss.item()


def train(model, train_loader, loss_function, optimizer, epochs, device, mps_fallback=False):
    model = model.to(device)
    model.train()

    print('[INFO] Launching training...\n')

    pbar_ep = tqdm(range(epochs), desc='[TQDM] Epoch #1 ', position=0, leave=False)

    for epoch in pbar_ep:

        total_train_loss, total_val_loss = 0, 0

        pbar_dl = tqdm(enumerate(train_loader), desc='[TQDM] Training on track 1/? ', total=train_loader.n_subtracks)

        last_track_idx = 0

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

            train_loss = loss_function(pred_edges, gt_edges, reduction='mean')

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            total_train_loss += train_loss.item()

            pbar_dl.update(1)

            """ Validation step """

            val_loader = train_loader
            val_loss = single_validate(model, val_loader, i + 1, loss_function, device)
            total_val_loss += val_loss

            """ Update progress """

            avg_train_loss_msg = f'avg.Tr.Loss: {(total_train_loss / (i + 1)):.4f} (last: {train_loss:.4f})'
            avg_val_loss_msg = f'avg.Val.Loss: {(total_val_loss / (i + 1)):.4f} (last: {val_loss:.4f})'

            pbar_ep.set_description(
                f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg} - {avg_val_loss_msg}')

            last_track_idx = cur_track_idx

        pbar_ep.set_description(f'[TQDM] Epoch #{epoch + 1} - avg.Loss: {(total_train_loss / (i + 1)):.4f}')


# %% Set up parameters

# Paths
# mot_path = 'data'
mot_path = '/media/dmmp/vid+backup/Data'

# MOT to use
mot = 'MOT17'

# Dtype to use
dtype = torch.float32

# Hyperparameters
backbone = 'resnet50'
layer_type = 'TransformerConv'
subtrack_len = 10
slide = 5
linkage_window = 5
l_size = 128
epochs = 1
learning_rate = 0.001

# Only if using MPS
mps_fallback = False

# %% Initialize the model

model = Net(backbone=backbone,
            layer_tipe=layer_type,
            layer_size=l_size,
            dtype=dtype,
            mps_fallback=True if device == torch.device('mps') else False,
            edge_dim=2,
            heads=3,
            concat=False,
            dropout=0.2,
            add_self_loops=False
            )

# TODO: train - val in different videos

# loss_function = torch.nn.BCEWithLogitsLoss() # TODO: look at focal loss to deal with the imbalance
loss_function = sigmoid_focal_loss
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
                          dtype=dtype,
                          black_and_white_features=False)

# %% Train the model

train(model, mot_train_dl, loss_function, optimizer, epochs, device, mps_fallback=mps_fallback)
