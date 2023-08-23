import argparse
import warnings

from torch_geometric.transforms import ToDevice

from diff_model import ImgEncoder, IMPLEMENTED_MODELS, Net
from diff_motclass import MotDataset
from puzzle_diff.model.spatial_diffusion import *

from utilities import *
from utilities import get_best_device
from diff_test import test

warnings.filterwarnings("ignore")


# %% Function definitions

def validation(model,
               val_loader,
               loss_function,
               device):
    """
    Wrapper around test function, used for validation.
    Will skip the tracks which are not chosen for validation.
    """
    return test(validation_mode=True, **locals())


def train(model,
          train_loader,
          val_loader,
          loss_function,
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

    average_train_loss, average_val_loss = 1e3, 1e3

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

        avg_train_loss_msg, avg_val_loss_msg, val_accs_msg = "", "", ""  # Useful to initialize for a nicer TQDM

        j = 0

        for _, data in pbar_dl:

            data = ToDevice(device.type)(data)

            cur_track_idx = train_loader.cur_track + 1
            cur_track_name = train_loader.tracklist[train_loader.cur_track]

            # IF VALIDATION TRACK THEN IGNORE
            if (cur_track_name in MOT17_VALIDATION_TRACKS) or (cur_track_name in MOT20_VALIDATION_TRACKS):
                pbar_dl.set_description(
                    f'[TQDM] Skipping track {cur_track_idx}/{len(train_loader.tracklist)} ({cur_track_name})')
                continue

            pbar_dl.set_description(
                f'[TQDM] Training on track {cur_track_idx}/{len(train_loader.tracklist)} ({cur_track_name})')

            """ Training step """

            # One-hot encoded y
            oh_y = torch.nn.functional.one_hot(data.y.to(torch.int64), -1)

            # Diffusion times
            time = torch.zeros((oh_y.shape[0])).to(device).to(torch.int64)

            # Edge attributes
            edge_attr = data.edge_attr

            # Edge indexes
            edge_index = data.edge_index

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

            total_train_loss += train_loss.item()

            average_train_loss = total_train_loss / (j + 1)

            epoch_info['avg_train_losses'].append(average_train_loss)

            avg_train_loss_msg = f'avg.Tr.Loss: {average_train_loss:.4f} (last: {train_loss:.4f})'

            pbar_ep.set_description(f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg}{avg_val_loss_msg}{val_accs_msg}')

            j += 1

        pbar_ep.set_description(
            f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg}{avg_val_loss_msg}{val_accs_msg}')

        """ Validation """
        # validate and save every 10 epochs
        if not epoch % 10 == 0:
            continue
        val_loss, acc_ones, acc_zeros, zeros_as_ones, ones_as_zeros = validation(model=model,
                                                                                 val_loader=val_loader,
                                                                                 loss_function=loss_function,
                                                                                 device=device)
        epoch_info['avg_val_losses'].append(val_loss)
        epoch_info['avg_accuracy_on_1'].append(acc_ones)
        epoch_info['avg_accuracy_on_0'].append(acc_zeros)
        epoch_info['avg_error_on_1'].append(ones_as_zeros)
        epoch_info['avg_error_on_0'].append(zeros_as_ones)
        epoch_info['tracklets'].append(j)

        avg_val_loss_msg = f' |  avg.Val.Loss: {val_loss:.4f})'

        val_accs_msg = f" - Accs: " \
                       f"[ 0 ✔ {acc_zeros :.2f} ] [ 1 ✔ {acc_ones :.2f}] " \
                       f"[ 0 ✖ {zeros_as_ones :.2f} ] [ 1 ✖ {ones_as_zeros:.2f} ]"

        pbar_ep.set_description(
            f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg}{avg_val_loss_msg}{val_accs_msg}')

        save_model(model,
                   mps_fallback=mps_fallback,
                   classification=classification,
                   epoch=epoch,
                   epoch_info=epoch_info,
                   node_model_name=model.model.model_dict['node_name'],
                   edge_model_name=model.model.model_dict['edge_name'],
                   savepath_adds={'trained_on': mot_train,
                                  "CIRO":'DIFFUSION'}# TODO: remove
                   )


# %% CLI args parser

parser = argparse.ArgumentParser(
    prog='python train.py',
    description='Script for training a graph network on the MOT task, integrating Diffusion.',
    epilog='Es: python diff_train.py',
    formatter_class=argparse.RawTextHelpFormatter)

# TODO: remove the default option before deployment
parser.add_argument('-D', '--datapath', default="/media/dmmp/vid+backup/Data",
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
parser.add_argument('-B', '--backbone', default="efficientnet_v2_l",
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
                         "Implemented losses: huber, l1, l2")
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
parser.add_argument('-l', '--learning_rate', default=0.001, type=float,
                    help="Learning rate.")
parser.add_argument('-b', '--diff-steps', default=100, type=int, # todo: change
                    help="Number of steps of the Diffusion process.")
parser.add_argument('--dropout', default=0.3, type=float,
                    help="Dropout probability.")
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

args = parser.parse_args()

classification = True  # in the Diffusion scenario should always be True

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
mps_fallback = args.apple_silicon  # args.apple_silicon  # Only if using MPS this should be true

# Diffusion steps
diffusion_steps = args.diff_steps

# Learning rate
learning_rate = args.learning_rate

# Loss function
loss_type = args.loss_function
match loss_type:
    case 'huber':
        loss_function = F.smooth_l1_loss
    case 'l1':
        loss_function = F.l1_loss
    case 'l2':
        loss_function = F.mse_loss
    case _:
        raise NotImplemented(
            "The chosen loss: " + loss_type + " has not been implemented yet."
                                              "To see the available ones, run this script with the -h option")

# %% Initialize the Diffusion model and the GNN backbone

args.model = "timeaware"

network_dict = IMPLEMENTED_MODELS[args.model]

gnn = Net(
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
    diff_steps=diffusion_steps,
    device=device,
    model_dict=network_dict,
    node_features_dim=ImgEncoder.output_dims[backbone],
    is_edge_model=not args.node_model,
    used_backbone=backbone)

model = GNN_Diffusion(custom_gnn=gnn,
                      steps=diffusion_steps,
                      mps_fallback=True).to(device)

optimizer = Adafactor(model.parameters(), lr=learning_rate, relative_step=False)

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
                          preprocessed=True,  # !
                          mps_fallback=mps_fallback,
                          classification=classification,
                          feature_extraction_backbone= backbone)

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
                        preprocessed=True,  # !
                        classification=classification,
                        mps_fallback=mps_fallback,
                        feature_extraction_backbone= backbone)

# Print information
print("[INFO] Hyperparameters and info:")
print("\n- Datasets:")
print("\t- Dataset used for training: " + mot_train + " | validation: " + mot_val)
print("\t- Subtrack length: " + str(subtrack_len))
print("\t- Linkage window: " + str(linkage_window))
print("\t- Sliding window: " + str(slide))
print("\t- Setting: classification") if classification else print("\t- Setting: regression")
print("\n- GNN backbone:")
print("\t- Backbone: " + backbone)
print("\t- Layer type: " + layer_type)
print("\t- Layer size: " + str(l_size))
print("\t- Heads: " + str(heads))
print("\t- Messages: " + str(messages))
print("\t- Dropout: " + str(args.dropout))
print("\t- Loss function: " + loss_type)
print("\n- Using Diffusion with " + str(diffusion_steps) + " steps")
print("")

# %% Train the model

train(model, mot_train_dl, mot_val_dl, loss_function, optimizer, epochs, device, mps_fallback)
