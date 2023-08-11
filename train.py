import argparse
import math
import random

import numpy
import torch

from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from model import Net, IMPLEMENTED_MODELS, ImgEncoder
from motclass import MotDataset
from utilities import *

# %% Function definitions

torch.autograd.set_detect_anomaly(True)

def single_validate(model, val_loader, idx, loss_function, device,loss_not_initialized=False, classification=False,**loss_arguments):
    """ Validate the model on a single subtrack, given a MOT dl and an index"""
    model.eval()

    data = val_loader[idx]

    with torch.no_grad():
        data = ToDevice(device.type)(data)
        pred_edges = model(data)  # Get the predicted edge labels
        gt_edges = data.y  # Get the true edge labels

        if loss_not_initialized:
            loss = loss_function(pred_edges, gt_edges, **loss_arguments)
        else:
            loss = loss_function(pred_edges, gt_edges)

        zero_treshold = 0.33
        one_treshold = 0.5

        if classification:
            zero_mask = pred_edges <= one_treshold
            one_mask = pred_edges > one_treshold
        else:
            zero_mask = pred_edges < zero_treshold
            one_mask = pred_edges > one_treshold

        pred_edges = torch.where(one_mask, 1., pred_edges)
        pred_edges = torch.where(zero_mask, 0., pred_edges)

        if classification:
            zero_mask = gt_edges <= one_treshold
            one_mask = gt_edges > one_treshold
        else:
            zero_mask = gt_edges < zero_treshold
            one_mask = gt_edges > one_treshold

        acc_ones = torch.where(pred_edges[one_mask] == 1., 1., 0.).mean()
        acc_zeros = torch.where(pred_edges[zero_mask] == 0., 1., 0.).mean()
        ones_as_zeros = torch.where(pred_edges[one_mask] == 0., 1., 0.).mean()
        zeros_as_ones = torch.where(pred_edges[zero_mask] == 1., 1., 0.).mean()

        if any([loss.isnan().item(), acc_ones.isnan().item(), acc_zeros.isnan().item(), zeros_as_ones.isnan().item(), ones_as_zeros.isnan().item()]):
            raise Exception("saduygsauygfuyasfgbv")

        return loss.item(), acc_ones.item(), acc_zeros.item(), zeros_as_ones.item(), ones_as_zeros.item()




def train(model, train_loader, val_loader, loss_function, optimizer, epochs, device, mps_fallback=False,
          loss_not_initialized=True, alpha=.95,
          gamma=2, reduction='mean', classification=False
          ):
    model = model.to(device)
    model.train()

    print('[INFO] Launching training...\n')

    pbar_ep = tqdm(range(epochs), desc='[TQDM] Epoch #1 ', position=0, leave=False,
                   bar_format="{desc:<5}{percentage:3.0f}%|{bar}{r_bar}")

    average_train_loss, average_val_loss = 1e3, 1e3
    for epoch in pbar_ep:

        epoch_info = {
            'tracklets': [],
            'avg_train_losses' : [],
            'avg_val_losses'   : [],
            'avg_accuracy_on_1': [],
            'avg_accuracy_on_0': [],
            'avg_error_on_1'   : [],
            'avg_error_on_0'   : [],
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
                save_model(model, mps_fallback=mps_fallback, classification=classification, epoch=epoch,
                           track_name=cur_track_name, epoch_info= epoch_info, node_model_name=model.model_dict['node_name'],
                           edge_model_name=model.model_dict['edge_name'])

            """ Training step """

            pred_edges = model(data)  # Get the predicted edge labels
            gt_edges = data.y  # Get the true edge labels

            # focal loss is implemented differently from the others
            if loss_not_initialized:
                train_loss = loss_function(pred_edges, gt_edges, alpha=alpha, gamma=gamma, reduction=reduction)
            else:
                train_loss = loss_function(pred_edges, gt_edges)


            if torch.isnan(train_loss):
                # debug
                pred_edges = model(data)
                raise Exception("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                pass
            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, error_if_nonfinite=True)

            pbar_dl.update(1)
            id_validate = random.choice(range(0, val_loader.n_subtracks))
            """ Validation step """
            val_loss, acc_ones, acc_zeros, zeros_as_ones, ones_as_zeros = single_validate(model=model,
                                                                                          val_loader=val_loader, idx=id_validate,
                                                                                          loss_function=loss_function,
                                                                                          device=device,
                                                                                          loss_not_initialized=loss_not_initialized,
                                                                                          classification=classification,
                                                                                          alpha=alpha, gamma=gamma, reduction=reduction)

            if numpy.isnan(val_loss)  or numpy.isnan(total_val_loss):
                pass
            total_train_loss += train_loss.item()
            total_val_loss += val_loss
            total_0acc += acc_zeros
            total_0err += zeros_as_ones
            total_1acc += acc_ones
            total_1err += ones_as_zeros

            average_train_loss = total_train_loss / (i + 1)
            average_val_loss   = total_val_loss / (i + 1)
            average_0acc       = total_0acc / (i + 1)
            average_0err       = total_0err / (i + 1)
            average_1acc       = total_1acc / (i + 1)
            average_1err       = total_1err / (i + 1)

            epoch_info['tracklets'].append(i)
            epoch_info['avg_train_losses' ].append(average_train_loss)
            epoch_info['avg_val_losses'   ].append(average_val_loss  )
            epoch_info['avg_accuracy_on_1'].append(average_1acc      )
            epoch_info['avg_accuracy_on_0'].append(average_0acc     )
            epoch_info['avg_error_on_1'   ].append(average_1err      )
            epoch_info['avg_error_on_0'   ].append(average_0err      )


            """ Update progress """

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
    return average_train_loss, average_val_loss


# CLI args parser
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='python train.py',
    description='Script for training a graph network on the MOT task',
    epilog='Es: python train.py',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-D', '--datapath', default="/media/dmmp/vid+backup/Data", help="""Dataset path
NB: This project assumes a MOT dataset, this project has been tested with MOT17 and MOT20""")  # TODO: remove the default option before deployment
parser.add_argument('--model_savepath', default="saves/models", help="""Folder where models are loaded""")
parser.add_argument('--output_savepath', default="saves/outputs", help="""Folder where outputs are saved""")
parser.add_argument('-m', '--MOTtrain', default="MOT17", help="""MOT dataset on which the network is trained""")
parser.add_argument('-M', '--MOTvalidation', default="MOT17",
                    help="""MOT dataset on which the single validate is calculated""")
parser.add_argument('-B', '--backbone', default="resnet50", help="""Visual backbone for nodes feature extraction""")
parser.add_argument('--float16', action='store_true', help="""Whether to use half floats or not""")
parser.add_argument('--apple', action='store_true', help="""Whether an apple device is in use with mps
(required for some fallbacks due to lack of mps support)""")
parser.add_argument('-Z', '--node_model', action='store_true', help="""Use the node model instead of the edge model""")
parser.add_argument('-L', '--loss_function', default="mse", help="""Loss function to use.
Implemented losses: huber, bce, focal, dice""")
parser.add_argument('--epochs', default=1, type=int, help="""Number of epochs
NB: one seems to be more than enough. The model is updated every new track and one epoch takes ~12h""")
parser.add_argument('-n', '--layer_size', default=500, type=int, help="""Size of hidden layers""")
parser.add_argument('-p', '--messages', default=6, type=int, help="""Number of message passing layers""")
parser.add_argument('--heads', default=6, type=int, help="""Number of heads, when applicable""")
parser.add_argument('--loss_reduction', default="mean", help="""Reduction logic for the loss
Implemented reductions: mean, sum""")
parser.add_argument('--model', default="timeaware", help="""Model to train
Implemented models: \n\ttimeaware, transformer, attention""")
parser.add_argument('--past_aggregation', default="mean", help="""Aggregation logic (past) for time aware
Implemented reductions: 'sum', 'add', 'mul', 'mean, 'min', 'max', 'std', 'logsumexp', 'softmax', 'log_softmax'""")
parser.add_argument('--future_aggregation', default="sum", help="""Aggregation logic (future) for time aware
Implemented reductions: 'sum', 'add', 'mul', 'mean, 'min', 'max', 'std', 'logsumexp', 'softmax', 'log_softmax'""")
parser.add_argument('--aggregation', default="mean", help="""Aggregation logic for other layers
Implemented reductions: 'mean', 'sum'""")
parser.add_argument('-l', '--learning_rate', default=0.0001, type=float, help="""learning rate""")
parser.add_argument('--dropout', default=0.3, type=float, help="""dropout""")
parser.add_argument('--optimizer', default="AdamW", help="""Optimizer to use
TODO: put available optimizers""")  # TODO
parser.add_argument('--alpha', default=0.95, type=float, help="""Alpha parameter for the focal loss
TODO: describe how it works""")  # TODO
parser.add_argument('--gamma', default=5., type=float, help="""Gamma parameter for the focal loss
TODO: describe how it works""")  # TODO
parser.add_argument('--delta', default=.1, type=float, help="""Delta parameter for the huber loss
TODO: describe how it works""")  # TODO
parser.add_argument('--detection_gt_folder', default="gt", help="""detection ground truth folder""")
parser.add_argument('--detection_gt_file', default="gt.txt", help="""detection ground truth folder""")
parser.add_argument('--subtrack_len', default=15, type=int, help="""Length of the subtrack
NB: a value higher than 20 might require too much memory""")
parser.add_argument('--linkage_window', default=5, type=int, help="""Linkage window for building the graph
es: w 5 -> detections in frame 0 will connect to detections up to frame 5""")
parser.add_argument('--slide', default=10, type=int, help="""Sliding window to adopt during testing
NB: suggested to be subtrack len - linkage window""")
parser.add_argument('-k', '--knn', default=20, type=int, help="""K parameter for knn reduction
NB: a value lower than 20 may exclude ground truths. Set to 0 for no knn""")
parser.add_argument('--cosine', action='store_true', help="""Use cosine distance instead of euclidean distance""")
parser.add_argument('--classification', action='store_true',
                    help="""Work in classification setting instead of regression""")
args = parser.parse_args()

# todo remove, used only for debug
# -------------------------------------------------------------------------------------------------------------------
# args.classification = True
# args.loss_function = "focal"
# args.model = 'transformer'
# -------------------------------------------------------------------------------------------------------------------


# there was no preconception of what to do  -cit.
classification = args.classification

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
subtrack_len = args.subtrack_len
slide = args.slide
linkage_window = args.linkage_window
messages = args.messages
l_size = args.layer_size
epochs = args.epochs
heads = args.heads
learning_rate = args.learning_rate
# Knn logic
if args.knn <= 0:
    knn_args = None
else:
    knn_args = {
        'k': args.knn,
        'cosine': args.cosine
    }

# Dtype to use
if args.float16:
    dtype = torch.float16
else:
    dtype = torch.float32

# Hyperparameters for graph logic
subtrack_len = args.subtrack_len
slide = args.slide
linkage_window = args.linkage_window

# device
device = get_best_device()
mps_fallback = args.apple  # Only if using MPS this should be true

# loss function
alpha = args.alpha
delta = args.delta
gamma = args.gamma
reduction = args.loss_reduction
loss_type = args.loss_function
loss_not_initialized = False
match loss_type:
    case 'huber':
        loss_function = IMPLEMENTED_LOSSES[loss_type](delta=delta, reduction=reduction)
        if classification:
            raise Exception("Huber loss should not be used in a classification setting, please choose a different one")
    case 'bce'| 'mae'| 'mse':
        loss_function = IMPLEMENTED_LOSSES[loss_type]()
    case 'focal':
        loss_function = IMPLEMENTED_LOSSES[loss_type]
        loss_not_initialized = True
    case 'berhu':
        raise NotImplemented("BerHu loss has not been implemented yet")
    case _:
        raise NotImplemented(
            "The chosen loss: " + loss_type + " has not been implemented yet. To see the available ones, run this script with the -h option")



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
                          knn_pruning_args=knn_args,
                          device=device,
                          dtype=dtype,
                          mps_fallback=mps_fallback,
                          classification=classification)

mot_val_dl = MotDataset(dataset_path=val_dataset_path,
                        split='train',
                        subtrack_len=subtrack_len,
                        slide=slide,
                        linkage_window=linkage_window,
                        detections_file_folder=detections_file_folder,
                        detections_file_name=detections_file_name,
                        knn_pruning_args=knn_args,
                        dl_mode=True,
                        device=device,
                        dtype=dtype,
                        mps_fallback=mps_fallback,
                        classification=classification)

network_dict = IMPLEMENTED_MODELS[args.model]

model = Net(backbone=backbone,
            layer_size=l_size,
            dtype=dtype,
            mps_fallback=mps_fallback,
            edge_features_dim=EDGE_FEATURES_DIM,
            heads=heads,
            concat=False,
            dropout=args.dropout,
            add_self_loops=False,
            steps=messages,
            device=device,
            model_dict=network_dict,
            node_features_dim=ImgEncoder.output_dims[backbone],
            is_edge_model=not args.node_model)

# %% Initialize the model

model.train()

optimizer = args.optimizer
optimizer = AVAILABLE_OPTIMIZERS[optimizer](model.parameters(), lr=learning_rate)


# print info
print("[INFO] hyper parameters:")
print("\nDatasets:")
print("\tDataset used for training: " + mot_train + " | validation: " + mot_val)
print("\tSubtrack lenght: " + str(subtrack_len) + "\n\t" +
      "Linkage window: " + str(linkage_window) + "\n\t" +
      "Slide: " + str(slide) + "\n\t" +
      "Knn : " + str(knn_args['k']))
if classification:
    print("\tSetting: classification")
else:
    print("\tSetting: regression")
print("\nNetwork:")
print("\tbackbone: " + backbone + "\n\t" +
      "number of heads: " + str(heads) + "\n\t" +
      "number of message passing steps: " + str(messages) + "\n\t" +
      "layer type: " + args.model + "\n\t" +
      "layer size: " + str(l_size) + "\n\t" +
      "number of edge features: " + str(EDGE_FEATURES_DIM) + "\n\t" +
      "dropout: " + str(args.dropout) + "\n"
      )
print("\n Prediction based on:")
if args.node_model:
    print('\tNodes\n')
else:
    print('Edges\n')
print("Training:")
print("\tLoss function: " + loss_type)
print("\tOptimizer: " + args.optimizer)
print("\tLearning rate: " + str(learning_rate))

# %% Train the model

train_loss, val_loss = train(model, mot_train_dl, mot_val_dl, loss_function, optimizer, epochs, device, mps_fallback,
      loss_not_initialized=loss_not_initialized, alpha=alpha,
      gamma=gamma, reduction=reduction, classification=classification
      )


print("\nFinal train loss: " + str(train_loss))

print("\nFinal validation loss: " + str(val_loss))