import argparse
import random

import numpy.random
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from model import Net
from motclass import MotDataset
from utilities import *
import torch
import pickle

from motclass import MotDataset
from puzzle_diff.model.spatial_diffusion import *
from utilities import get_best_device
from torch_geometric.transforms import ToDevice

from model import ImgEncoder, IMPLEMENTED_MODELS


# %% Function definitions

def train(model,
          train_loader,
          val_loader,
          optimizer,
          epochs,
          device,
          mps_fallback=False):
    """
    Main training function.
    """
    model = model.to(device)
    model.train()

    print('[INFO] Launching training...\n')

    pbar_ep = tqdm(range(epochs), desc='[TQDM] Epoch #1 ', position=0, leave=False,
                   bar_format="{desc:<5}{percentage:3.0f}%|{bar}{r_bar}")

    for epoch in pbar_ep:

        epoch_info = {
            'tracklets': [],
            'avg_train_losses': [],
            'avg_val_losses': [],
            'avg_accuracy_on_1': [],
            'avg_accuracy_on_0': [],
            'avg_error_on_1': [],
            'avg_error_on_0': [],
        }

        total_train_loss, total_val_loss, total_1acc, total_0acc, total_1err, total_0err = 0, 0, 0, 0, 0, 0

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
            if cur_track_idx != last_track_idx and cur_track_idx != 1:
                save_model(model, mps_fallback=mps_fallback, classification=classification, epoch=epoch, epoch_info=epoch_info,
                           savepath_adds={'extra': 'diffusion'})

            # -------------------------------------------------------------------------------------------------------------------- #

            " Training step "

            # One-hot encoded y - INVERSE!
            oh_y = torch.nn.functional.one_hot(data.y.to(torch.int64), -1)  # .to(device).float()

            # Diffusion times
            time = torch.zeros((oh_y.shape[0])).to(device).to(torch.int64)

            # Edge attributes
            edge_attr = data.edge_attr  # .to(device)

            # Edge indexes
            edge_index = data.edge_index  # .to(device)

            train_loss = model.p_losses(
                x_start=oh_y,
                t=time,
                loss_type="huber",
                node_feats=data.detections,
                edge_index=edge_index,
                edge_feats=edge_attr
            )

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            pbar_dl.update(1)

            # -------------------------------------------------------------------------------------------------------------------- #

            " Validation step "
            id_validate = random.choice(range(0, val_loader.n_subtracks))
            val_loss, acc_ones, acc_zeros, zeros_as_ones, ones_as_zeros = single_validate(model=model,
                                                                                          val_loader=val_loader,
                                                                                          idx=id_validate, # todo
                                                                                          device=device)

            total_train_loss += train_loss.item()
            total_val_loss += val_loss
            total_0acc += acc_zeros
            total_0err += zeros_as_ones
            total_1acc += acc_ones
            total_1err += ones_as_zeros

            average_train_loss = total_train_loss / (i + 1)
            average_val_loss = total_val_loss / (i + 1)
            average_0acc = total_0acc / (i + 1)
            average_0err = total_0err / (i + 1)
            average_1acc = total_1acc / (i + 1)
            average_1err = total_1err / (i + 1)

            epoch_info['tracklets'].append(i)
            epoch_info['avg_train_losses'].append(average_train_loss)
            epoch_info['avg_val_losses'].append(average_val_loss)
            epoch_info['avg_accuracy_on_1'].append(average_0acc)
            epoch_info['avg_accuracy_on_0'].append(average_0err)
            epoch_info['avg_error_on_1'].append(average_1acc)
            epoch_info['avg_error_on_0'].append(average_1err)

            " Update progress "

            avg_train_loss_msg = f'avg.Tr.Loss: {average_train_loss:.4f} (last: {train_loss:.4f})'
            avg_val_loss_msg = f'avg.Val.Loss: {average_val_loss:.4f} (last: {val_loss:.4f})'

            avg_0acc_msg = f'avg. 0 acc: {average_0acc:.2f} (last: {acc_zeros:.2f})'
            avg_1acc_loss_msg = f'avg. 1 acc: {average_1acc:.2f} (last: {acc_ones:.2f})'
            avg_10_loss_msg = f'avg. 1 as 0: {average_1err:.2f} (last: {ones_as_zeros:.2f})'
            avg_01_loss_msg = f'avg. 0 as 1: {average_0err:.2f} (last: {zeros_as_ones:.2f})'

            pbar_ep.set_description(
                f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg} - {avg_val_loss_msg} - {avg_0acc_msg} - {avg_1acc_loss_msg} - {avg_10_loss_msg} - {avg_01_loss_msg}')

            last_track_idx = cur_track_idx

        pbar_ep.set_description(f'[TQDM] Epoch #{epoch + 1} - avg.Loss: {(total_train_loss / (i + 1)):.4f}')


def single_validate(model,
                    val_loader,
                    idx,
                    device):
    """
    Validate the model on a single subtrack, given a MOT dataloader and an index.
    """
    model.eval()

    data = val_loader[idx]

    with torch.no_grad():
        data = ToDevice(device.type)(data)

        gt_edges = data.y

        # One-hot encoded y - INVERSE!
        oh_y = torch.nn.functional.one_hot(data.y.to(torch.int64), -1)  # .to(device).float()

        # Diffusion times TODO: check time use
        time = torch.zeros((oh_y.shape[0])).to(device).to(torch.int64)

        # Edge attributes
        edge_attr = data.edge_attr  # .to(device)

        # Edge indexes
        edge_index = data.edge_index  # .to(device)

        _, pred_edges_oh = model.p_sample_loop(shape=(oh_y.shape[0], 2),
                                               edge_feats=edge_attr,
                                               node_feats=data.detections,
                                               edge_index=edge_index)
        loss = F.smooth_l1_loss(pred_edges_oh, oh_y)
        # loss, _ = model.p_losses(
        #     x_start=oh_y,
        #     t=time,
        #     loss_type="huber",
        #     node_feats=data.detections,
        #     edge_feats=edge_attr,
        #     edge_index=edge_index,
        #     cond=None,
        #     also_preds=True
        # )

        # ths = (pred_edges_oh[:, 1] - pred_edges_oh[:, 0]).mean()

        # ths = 0
        # pred_edges = torch.where(pred_edges_oh[:, 1] > ths, 1., 0.)

        pred_edges = torch.where(pred_edges_oh[:, 1] > pred_edges_oh[:, 0], 1., 0.)  # !

        # apply a softmax to the output of the model
        # pred_edges = torch.softmax(pred_edges_oh, dim=1)[:, 1]
        # pred_edges = torch.round(pred_edges)

        zero_mask = gt_edges <= .5
        one_mask = gt_edges > .5

        acc_ones = torch.where(pred_edges[one_mask] == 1., 1., 0.).mean()
        acc_zeros = torch.where(pred_edges[zero_mask] == 0., 1., 0.).mean()
        ones_as_zeros = torch.where(pred_edges[one_mask] == 0., 1., 0.).mean()
        zeros_as_ones = torch.where(pred_edges[zero_mask] == 1., 1., 0.).mean()

        return loss.item(), acc_ones.item(), acc_zeros.item(), zeros_as_ones.item(), ones_as_zeros.item()


# %% CLI args parser

parser = argparse.ArgumentParser(
    prog='python train.py',
    description='Script for training a graph network on the MOT task',
    epilog='Es: python train.py',
    formatter_class=argparse.RawTextHelpFormatter)

# TODO: remove the default option before deployment
parser.add_argument('-D', '--datapath', default="data",
                    help="Path to the folder containing the MOT datasets."
                         "NB: This project assumes a MOT dataset, this project has been tested with MOT17 and MOT20")
parser.add_argument('--model_savepath', default="saves/models",
                    help="Folder where models are loaded.")
parser.add_argument('--output_savepath', default="saves/outputs",
                    help="Folder where outputs are saved.")
parser.add_argument('-m', '--MOTtrain', default="MOT17",
                    help="MOT dataset on which the network is trained.")
parser.add_argument('-M', '--MOTvalidation', default="MOT20",
                    help="MOT dataset on which the single validate is calculated.")
parser.add_argument('-N', '--message_layer_nodes', default="base",
                    help="Type of message passing layer for nodes (NODE MODEL)."
                         "NB: all layers are time aware"
                         "Available structures:\n"
                         "- base (layer proposed on Neural Solver),\n"
                         "- attention (GATv2Conv),\n"
                         "- transformer")
# parser.add_argument('-E', '--message_layer_edges', default="base",
#                     help="Type of message passing layer for edges (EDGE MODEL)."
#                          "NB: all layers are time aware"
#                          "Available structures:\n"
#                          "- base (layer proposed on Neural Solver),\n"
#                          "- general (GeneralConv),\n"
#                          "- GAT (GATv2Conv),\n"
#                          "- transformer")
parser.add_argument('-B', '--backbone', default="resnet50",
                    help="Visual backbone for nodes feature extraction.")
parser.add_argument('--float16', action='store_true',
                    help="Whether to use half floats or not.")
parser.add_argument('--apple-silicon', action='store_true',
                    help="Whether a Mac with Apple Silicon is in use with MPS acceleration."
                         "(required for some fallbacks due to lack of MPS support)")
parser.add_argument('-Z', '--node_model', action='store_true',
                    help="Use the node model instead of the edge model.")
parser.add_argument('-L', '--loss_function', default="huber",
                    help="Loss function to use."
                         "Implemented losses: huber, bce, focal, dice")
parser.add_argument('--epochs', default=1, type=int,
                    help="Number of epochs.")
parser.add_argument('-n', '--layer_size', default=500, type=int,
                    help="Size of hidden layers")
parser.add_argument('-p', '--messages', default=6, type=int,
                    help="Number of message passing layers")
parser.add_argument('--heads', default=6, type=int,
                    help="Number of heads, when applicable")
parser.add_argument('--reduction', default="mean",
                    help="Reduction logic for the loss."
                         "Implemented reductions: mean, sum")
# parser.add_argument('-l', '--learning_rate', default=0.001, type=float,
#                     help="Learning rate.")
parser.add_argument('--dropout', default=0.3, type=float, help="dropout")
# TODO: put available optimizers
# parser.add_argument('--optimizer', default="AdamW",
#                     help="Optimizer to use.")
# TODO: describe how it works
parser.add_argument('--alpha', default=0.95, type=float,
                    help="Alpha parameter for the focal loss.")
# TODO: describe how it works
parser.add_argument('--gamma', default=2., type=float,
                    help="Gamma parameter for the focal loss.")
# TODO: describe how it works
parser.add_argument('--delta', default=.4, type=float,
                    help="Delta parameter for the huber loss.")
parser.add_argument('--detection_gt_folder', default="gt",
                    help="Folder containing the ground truth detections files.")
parser.add_argument('--detection_gt_file', default="gt.txt",
                    help="Name of the ground truth detections file.")
parser.add_argument('--subtrack_len', default=15, type=int,
                    help="Length of the subtrack."
                         "NB: a value higher than 20 might require too much memory.")
parser.add_argument('--linkage_window', default=5, type=int,
                    help="Linkage window for building the graph."
                         "(e.s. if = 5 -> detections in frame 0 will connect to detections up to frame 5)")
parser.add_argument('--slide', default=10, type=int,
                    help="Sliding window to adopt during testing."
                         "NB: suggested to be subtrack_len - linkage_window")
# parser.add_argument('-k', '--knn', default=20, type=int,
#                     help="K parameter for knn reduction."
#                          "NB: a value lower than 20 may exclude ground truths. Set to 0 for no knn.")
# parser.add_argument('--cosine', action='store_true',
#                     help="Use cosine distance instead of euclidean distance. Unavailable under MPS.")
# parser.add_argument('--classification', action='store_true',
#                     help="Work in classification setting instead of regression.")

args = parser.parse_args()

# There was no preconception of what to do  -cit.
classification = True

# %% Set up parameters

# Paths
mot_path = args.datapath
detections_file_folder = args.detection_gt_folder
detections_file_name = args.detection_gt_file

# MOT to use
mot_train = args.MOTtrain
mot_val = args.MOTvalidation

# Hyperparameters
backbone = args.backbone
layer_type = args.message_layer_nodes
subtrack_len = args.subtrack_len
slide = args.slide
linkage_window = args.linkage_window
messages = args.messages
l_size = args.layer_size
epochs = args.epochs
heads = args.heads
# learning_rate = args.learning_rate TODO

# Dtype to use
if args.float16:
    dtype = torch.float16
else:
    dtype = torch.float32

# Hyperparameters for graph logic
subtrack_len = args.subtrack_len
slide = args.slide
linkage_window = args.linkage_window

# Device
device = get_best_device()
mps_fallback = args.apple_silicon # args.apple_silicon  # Only if using MPS this should be true

# Loss function
# alpha = args.alpha
# delta = args.delta
# gamma = args.gamma
# reduction = args.reduction
# loss_type = args.loss_function
# match loss_type:
#     case 'huber':
#         loss_function = IMPLEMENTED_LOSSES[loss_type](delta=delta, reduction=reduction)
#         if classification:
#             raise Exception("Huber loss should not be used in a classification setting, please choose a different one")
#     case 'bce':
#         loss_function = IMPLEMENTED_LOSSES[loss_type]()
#     case 'focal':
#         loss_function = IMPLEMENTED_LOSSES[loss_type]
#     case 'berhu':
#         raise NotImplemented("BerHu loss has not been implemented yet")
#     case _:
#         raise NotImplemented(
#             "The chosen loss: " + loss_type + " has not been implemented yet."
#                                               "To see the available ones, run this script with the -h option")
#
# %% Initialize the model

args.model = "timeaware"

network_dict = IMPLEMENTED_MODELS[args.model]

gnn = Net(backbone=backbone,
          layer_tipe=layer_type,
          layer_size=l_size,
          dtype=dtype,
          mps_fallback=mps_fallback,
          edge_features_dim=70,
          heads=heads,
          concat=False,
          dropout=args.dropout,
          add_self_loops=False,
          steps=messages,
          device=device,
          model_dict=network_dict,
          node_features_dim=ImgEncoder.output_dims[backbone],
          is_edge_model=not args.node_model)

model = GNN_Diffusion(custom_gnn=gnn, mps_fallback=mps_fallback).to(device)

optimizer = Adafactor(model.parameters())

# %% Set up the dataloader

train_dataset_path = os.path.normpath(os.path.join(mot_path, mot_train))
val_dataset_path = os.path.normpath(os.path.join(mot_path, mot_val))

mot_train_dl = MotDataset(dataset_path=train_dataset_path,
                          split='train',
                          subtrack_len=subtrack_len,
                          slide=slide,
                          linkage_window=linkage_window,
                          detections_file_folder=detections_file_folder,
                          detections_file_name=detections_file_name,
                          dl_mode=True,
                          device=device,
                          dtype=dtype,
                          mps_fallback=mps_fallback,
                          preprocessed=True,
                          classification=classification)

mot_val_dl = MotDataset(dataset_path=val_dataset_path,
                        split='train',
                        subtrack_len=subtrack_len,
                        slide=slide,
                        linkage_window=linkage_window,
                        detections_file_folder=detections_file_folder,
                        detections_file_name=detections_file_name,
                        dl_mode=True,
                        device=device,
                        dtype=dtype,
                        preprocessed=True,
                        mps_fallback=mps_fallback)

# Print information
print("[INFO] hyper parameters:")
print("\nDatasets:")
print("\tDataset used for training: " + mot_train + " | validation: " + mot_val)
print("\tSubtrack lenght: " + str(subtrack_len) + "\n\t" +
      "Linkage window: " + str(linkage_window) + "\n\t" +
      "Slide: " + str(slide) + "\n\t" )

print("\tSetting: classification")


# %% Train the model

train(model, mot_train_dl, mot_val_dl, optimizer, epochs, device, mps_fallback)
