import argparse
import warnings

from torch_geometric.transforms import ToDevice
from torchvision.ops import sigmoid_focal_loss

from diff_model import *
from diff_motclass import MotDataset
from diff_test import test
from puzzle_diff.model.spatial_diffusion import *
from utilities import *

warnings.filterwarnings("ignore")

device = get_best_device()


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
          n_vals,
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

    pbar_ep = tqdm(range(1, epochs + 1), desc='[TQDM] Epoch #1 ', position=0, leave=False,
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
            time = torch.randint(0, args.diff_steps, (oh_y.shape[0],)).to(device).to(torch.int64)

            # Edge attributes
            edge_attr = data.edge_attr

            # Edge indexes
            edge_index = data.edge_index

            train_loss = model.p_losses(
                x_start=oh_y,
                t=time,
                loss_type=args.loss_function,
                node_feats=data.detections,
                edge_index=edge_index,
                edge_feats=edge_attr,
            )

            pass

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, error_if_nonfinite=True)

            total_train_loss += train_loss.item()

            average_train_loss = total_train_loss / (j + 1)

            epoch_info['avg_train_losses'].append(average_train_loss)

            avg_train_loss_msg = f'avg.Tr.Loss: {average_train_loss:.6f} (last: {train_loss:.6f})'

            pbar_ep.set_description(f'[TQDM] Epoch #{epoch} - {avg_train_loss_msg}{avg_val_loss_msg}{val_accs_msg}')

            j += 1

        pbar_ep.set_description(
            f'[TQDM] Epoch #{epoch} - {avg_train_loss_msg}{avg_val_loss_msg}{val_accs_msg}')

        """ Validation """

        # Only validate n_vals times throughout the training
        if not epoch % (epochs // n_vals) == 0:
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

        save_model(model,
                   mps_fallback=mps_fallback,
                   classification=True,
                   epoch=epoch,
                   epoch_info=epoch_info)

        avg_val_loss_msg = f' |  avg.Val.Loss: {val_loss:.6f})'

        val_accs_msg = f" - Accs: " \
                       f"[ 0 ✔ {acc_zeros :.2f} ] [ 1 ✔ {acc_ones :.2f}] " \
                       f"[ 0 ✖ {zeros_as_ones :.2f} ] [ 1 ✖ {ones_as_zeros:.2f} ]"

        pbar_ep.set_description(
            f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg}{avg_val_loss_msg}{val_accs_msg}')


# %% CLI args parser

parser = argparse.ArgumentParser(
    prog='python train.py',
    description='Script for training a graph network on the MOT task, integrating Diffusion.',
    epilog='Es: python diff_train.py',
    formatter_class=argparse.RawTextHelpFormatter)

# ---------------- Paths --------------------------------------------------------------------------------------------- #

parser.add_argument('-D', '--datapath', default="data",  # TODO: remove default
                    help="Path to the folder containing the MOT datasets."
                         "NB: This project assumes a MOT dataset, this project has been tested with MOT17 and MOT20")

parser.add_argument('--model-savepath', default="saves/models",
                    help="Folder where models are loaded.")

parser.add_argument('--output-savepath', default="saves/outputs",
                    help="Folder where outputs are saved.")

# ---------------- Training and validation logics -------------------------------------------------------------------- #

parser.add_argument('-m', '--MOTtrain', default="MOT17",
                    help="MOT dataset on which the network is trained.")

parser.add_argument('-M', '--MOTvalidation', default="MOT17",
                    help="MOT dataset on which the single validate is calculated.")

parser.add_argument('--n-vals', default=3, type=int,
                    help="Number of validation steps to execute during training."
                         "(Validation can be quite expensive, and the model learns slow)")

parser.add_argument('--detections-file-folder', default="gt",
                    help="Folder containing the detections file.")

parser.add_argument('--detections-file', default="gt.txt",
                    help="Name of the actual detections file.")

# ---------------- Main training parameters -------------------------------------------------------------------------- #

parser.add_argument('-L', '--loss-function', default="focal",
                    help="Loss function to use."
                         "Implemented losses: focal, huber, l1, l2")

parser.add_argument('--epochs', default=30, type=int,
                    help="Number of epochs.")

parser.add_argument('-l', '--learning-rate', default=0.001, type=float,
                    help="Learning rate.")

# ---------------- Architecture choices ------------------------------------------------------------------------------ #

parser.add_argument('-N', '--mp-arch', default="base",
                    help="Type of message passing architecture."
                         "NB: all layers are time aware"
                         "Available structures:\n"
                         "- base (layer proposed on Neural Solver),\n"
                         "- attention (GATv2Conv),\n"
                         "- transformer (TransformerConv)")

parser.add_argument('-B', '--backbone', default="efficientnet_v2_l",
                    help="Visual backbone for nodes feature extraction.")

parser.add_argument('-n', '--layer-size', default=128, type=int,
                    help="Size of hidden layers")

parser.add_argument('-p', '--messages', default=6, type=int,
                    help="Number of message passing layers.")

parser.add_argument('--heads', default=6, type=int,
                    help="Number of heads, when applicable.")

parser.add_argument('--reduction', default="mean",
                    help="Reduction logic for the loss."
                         "Implemented reductions: mean, sum")

parser.add_argument('-Z', '--node-model', action='store_true',
                    help="Use the node model instead of the edge model.")

parser.add_argument('--dropout', default=0.3, type=float,
                    help="Dropout probability.")

# ---------------- Diffusion parameters ------------------------------------------------------------------------------ #

parser.add_argument('-b', '--diff-steps', default=300, type=int,
                    help="Number of steps of the Diffusion process.")

# ---------------- Graph building parameters ------------------------------------------------------------------------- #

parser.add_argument('--subtrack-len', default=15, type=int,
                    help="Length of the subtrack."
                         "NB: a value higher than 20 might require too much memory.")

parser.add_argument('--linkage-window', default=5, type=int,
                    help="Linkage window for building the graph."
                         "(e.s. if = 5 -> detections in frame 0 will connect to detections up to frame 5)")

parser.add_argument('--slide', default=10, type=int,
                    help="Sliding window to adopt during testing."
                         "NB: suggested to be subtrack_len - linkage-window")

# ---------------- Use preprocessed data ----------------------------------------------------------------------------- #

parser.add_argument('--train-preprocessed', action='store_true', default=True,
                    help="Whether to use preprocessed features for train dataloader.")

parser.add_argument('--val-preprocessed', action='store_true', default=True,
                    help="Whether to use preprocessed features for val dataloader.")

# ----------------- Miscellaneous ------------------------------------------------------------------------------------ #

parser.add_argument('--float16', action='store_true',
                    help="Whether to use half floats or not.")

parser.add_argument('--apple-silicon', action='store_true',
                    help="Whether a Mac with Apple Silicon is in use with MPS acceleration."
                         "(required for some fallbacks due to lack of MPS support)")

# -------------------------------------------------------------------------------------------------------------------- #

args = parser.parse_args()

# %% Set up Loss function

match args.loss_function:
    case 'focal':
        loss_function = sigmoid_focal_loss
    case 'huber':
        loss_function = F.smooth_l1_loss
    case 'l1':
        loss_function = F.l1_loss
    case 'l2':
        loss_function = F.mse_loss
    case _:
        raise NotImplemented(
            "The chosen loss: " + args.loss_function + " is invalid or has not been implemented yet."
                                                       "To see the available ones, run this script with the -h option")

# %% Initialize the GNN and the Diffusion model

gnn = Net(
    layer_tipe=args.mp_arch,
    layer_size=args.layer_size,
    dtype=torch.float16 if args.float16 else torch.float32,
    mps_fallback=args.apple_silicon,
    edge_features_dim=70,
    heads=args.heads,
    concat=False,
    dropout=args.dropout,
    add_self_loops=False,
    steps=3,
    device=device,
    model_dict=IMPLEMENTED_MODELS[args.mp_arch],
    node_features_dim=ImgEncoder.output_dims[args.backbone],
    is_edge_model=not args.node_model,
    used_backbone=args.backbone,
    diff_steps=args.diff_steps
).to(device)

model = GNN_Diffusion(custom_gnn=gnn,
                      steps=args.diff_steps,
                      mps_fallback=args.apple_silicon).to(device)

optimizer = Adafactor(model.parameters(), lr=args.learning_rate, relative_step=False)

# %% Set up the dataloader

train_dataset_path = os.path.normpath(os.path.join(args.datapath, args.MOTtrain))
val_dataset_path = os.path.normpath(os.path.join(args.datapath, args.MOTvalidation))

mot_train_dl = MotDataset(dataset_path=train_dataset_path,
                          split='train',
                          subtrack_len=args.subtrack_len,
                          slide=args.slide,
                          linkage_window=args.linkage_window,
                          detections_file_folder=args.detections_file_folder,
                          detections_file_name=args.detections_file,
                          dl_mode=True,
                          device=device,
                          dtype=torch.float16 if args.float16 else torch.float32,
                          preprocessed=args.val_preprocessed,
                          mps_fallback=args.apple_silicon,
                          classification=True,
                          feature_extraction_backbone=args.backbone)

mot_val_dl = MotDataset(dataset_path=val_dataset_path,
                        split='train',
                        subtrack_len=args.subtrack_len,
                        slide=args.slide,
                        linkage_window=args.linkage_window,
                        detections_file_folder=args.detections_file_folder,
                        detections_file_name=args.detections_file,
                        dl_mode=True,
                        device=device,
                        dtype=torch.float16 if args.float16 else torch.float32,
                        preprocessed=args.train_preprocessed,
                        classification=True,
                        mps_fallback=args.apple_silicon,
                        feature_extraction_backbone=args.backbone)

# Check at which epochs to validate
val_at = [str(epoch) for epoch in range(1, args.epochs + 1) if epoch % (args.epochs // args.n_vals) == 0]

# Print information
print("[INFO] Hyperparameters and info:")
print("\n- Datasets:")
print("\t- Training on " + args.MOTtrain + "-train")
print("\t- Validating on " + args.MOTvalidation + "-val")
print("\t- Saving and validating at epochs " + ", ".join(val_at))
print("\t- Subtrack length: " + str(args.subtrack_len))
print("\t- Linkage window: " + str(args.linkage_window))
print("\t- Sliding window: " + str(args.slide))
print("\n- GNN backbone:")
print("\t- Visual backbone: " + args.backbone)
print("\t- MP architecture: " + args.mp_arch)
print("\t- Message passing steps: " + str(args.messages))
print("\t- Loss function: " + args.loss_function)
print("\n- Diffusion: ")
print("\t- Diffusion steps: " + str(args.diff_steps))
print("")

# %% Train the model

train(model,
      mot_train_dl,
      mot_val_dl,
      args.n_vals,
      loss_function,
      optimizer,
      args.epochs,
      device,
      args.apple_silicon)
