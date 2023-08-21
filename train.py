import argparse
import warnings

from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from model import Net, IMPLEMENTED_MODELS, ImgEncoder
from motclass import MotDataset
from test import test
from utilities import *

warnings.filterwarnings("ignore")  # Nice TQDMs >>>>>> blissful ignorance

# %% Function definitions

torch.autograd.set_detect_anomaly(True)


def validation(model,
               val_loader,
               loss_function,
               device):
    """
    Wrapper around test function, used for validation.
    Will skip the tracks which are not chosen for validation.
    """
    return test( validation_mode=True, **locals())


def train(model,
          train_loader,
          val_loader,
          loss_function,
          optimizer,
          epochs,
          device,
          mps_fallback=False,
          alpha=.95,
          gamma=2,
          reduction='mean',
          classification=False) -> (float, float):
    """
    Main training logic for the GNN model.
    Will train on all training tracks, and validate on all validation partition at the end of each track.
    """
    global mot_train

    model = model.to(device)
    model.train()

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

        pbar_dl = tqdm(enumerate(train_loader), desc='[TQDM]', total=train_loader.n_subtracks,
                       bar_format="{desc:<5}{percentage:3.0f}%|{bar}{r_bar}", leave=True)

        last_track_idx = 0

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

            pred_edges = model(data)  # Get the predicted edge labels
            gt_edges = data.y  # Get the true edge labels


            train_loss = loss_function(pred_edges, gt_edges)

            if torch.isnan(train_loss):
                raise Exception("Why are we still here? Just to suffer")

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, error_if_nonfinite=True)

            pbar_dl.update(1)

            total_train_loss += train_loss.item()

            average_train_loss = total_train_loss / (j + 1)

            epoch_info['avg_train_losses'].append(average_train_loss)

            avg_train_loss_msg = f'avg.Tr.Loss: {average_train_loss:.4f} (last: {train_loss:.4f})'

            pbar_ep.set_description(f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg}{avg_val_loss_msg}{val_accs_msg}')

            last_track_idx = cur_track_idx
            j += 1

        pbar_ep.set_description(
            f'[TQDM] Epoch #{epoch + 1} - {avg_train_loss_msg}{avg_val_loss_msg}{val_accs_msg}')

        # VAIDATION

        val_loss, acc_ones, acc_zeros, zeros_as_ones, ones_as_zeros = validation(model=model,
                                                                                 val_loader=val_loader,
                                                                                 loss_function= loss_function,
                                                                                 device= device)
        epoch_info['avg_val_losses'].append(val_loss)
        epoch_info['avg_accuracy_on_1'].append(acc_ones)
        epoch_info['avg_accuracy_on_0'].append(acc_zeros)
        epoch_info['avg_error_on_1'].append(ones_as_zeros)
        epoch_info['avg_error_on_0'].append(zeros_as_ones)

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
                   node_model_name=model.model_dict['node_name'],
                   edge_model_name=model.model_dict['edge_name'], savepath_adds={'trained_on':mot_train})

    return average_train_loss, val_loss


if __name__ == '__main__':

    # CLI args parser
    parser = argparse.ArgumentParser(
        prog='python train.py',
        description='Script for training a graph network on the MOT task',
        epilog='Es: python train.py',
        formatter_class=argparse.RawTextHelpFormatter)

    # TODO: remove the default option before deployment
    parser.add_argument('-D', '--datapath', default="/media/dmmp/vid+backup/Data",
                        help="Dataset path"
                             "NB: This project assumes a MOT dataset, this project has been tested with MOT17 and MOT20.")

    parser.add_argument('--model_savepath', default="saves/models",
                        help="Folder where models are loaded.")

    parser.add_argument('--output_savepath', default="saves/outputs",
                        help="Folder where outputs are saved.")

    parser.add_argument('-m', '--MOTtrain', default="MOT17",
                        help="MOT dataset on which the network is trained.")

    parser.add_argument('-M', '--MOTvalidation', default="MOT17",
                        help="MOT dataset on which the single validate is calculated.")

    parser.add_argument('-B', '--backbone', default="efficientnet_v2_l",
                        help="Visual backbone for nodes feature extraction.")

    parser.add_argument('--float16', action='store_true',
                        help="Whether to use half floats or not.")

    parser.add_argument('--apple-silicon', action='store_true',
                        help="Whether an apple device is in use with mps (required for some fallbacks due to lack of mps support).")

    parser.add_argument('-Z', '--node-model', action='store_true',
                        help="Use the node model instead of the edge model.")

    parser.add_argument('-L', '--loss-function', default="focal",
                        help="Loss function to use."
                             "Implemented losses: huber, bce, focal, dice.")

    parser.add_argument('--epochs', default=1, type=int,
                        help="Number of epochs.")

    parser.add_argument('-n', '--layer-size', default=500, type=int,
                        help="Size of hidden layers.")

    parser.add_argument('-p', '--messages', default=6, type=int,
                        help="Number of message passing layers.")

    parser.add_argument('--heads', default=6, type=int,
                        help="Number of heads, when applicable.")

    parser.add_argument('--loss-reduction', default="mean",
                        help="Reduction logic for the loss."
                             "Implemented reductions: mean, sum.")

    parser.add_argument('--model', default="attention",
                        help="Model to train Implemented models: timeaware, transformer, attention.")

    parser.add_argument('--past-aggregation', default="mean",
                        help="Aggregation logic (past) for time aware."
                             "Implemented reductions: 'sum', 'add', 'mul', 'mean, 'min', 'max', 'std', 'logsumexp', 'softmax', 'log_softmax'.")

    parser.add_argument('--future-aggregation', default="sum",
                        help="Aggregation logic (future) for time aware."
                             "Implemented reductions: 'sum', 'add', 'mul', 'mean, 'min', 'max', 'std', 'logsumexp', 'softmax', 'log_softmax'.")

    parser.add_argument('--aggregation', default="mean",
                        help="Aggregation logic for other layers."
                             "Implemented reductions: 'mean', 'sum'.")

    parser.add_argument('-l', '--learning-rate', default=0.0001, type=float,
                        help="Learning rate to use for the optimizer.")

    parser.add_argument('--dropout', default=0.2, type=float,
                        help="Dropout probability")

    # TODO: put available optimizers
    parser.add_argument('--optimizer', default="AdamW",
                        help="Optimizer to use.")

    # TODO: describe how it works
    parser.add_argument('--alpha', default=0.05, type=float,
                        help="Alpha parameter for the focal loss.")

    # TODO: describe how it works
    parser.add_argument('--gamma', default=5., type=float,
                        help="Gamma parameter for the focal loss")

    # TODO: describe how it works
    parser.add_argument('--delta', default=.3, type=float,
                        help="Delta parameter for the huber loss")

    parser.add_argument('--detection_folder', default="gt",
                        help="detection ground truth folder")

    parser.add_argument('--detection_file', default="gt.txt",
                        help="detection ground truth folder")


    parser.add_argument('--subtrack-len', default=15, type=int,
                        help="Length of the subtrack."
                             "NB: a value higher than 20 might require too much memory.")

    parser.add_argument('--linkage-window', default=5, type=int,
                        help="Linkage window for building the graph."
                             "E.g. with a window of 5 detections in frame 0 will connect to detections up to frame 5")

    parser.add_argument('--slide', default=10, type=int,
                        help="Sliding window to adopt during testing."
                             "NB: suggested to be subtrack len - linkage window")

    parser.add_argument('--classification', action='store_true',
                        help="Work in classification setting instead of regression.")
    parser.add_argument('--node_model', action='store_true',
                        help="node model")

    parser.add_argument('--train-preprocessed', action='store_true', default=True,
                        help="Directly use preprocessed data for training dataloader.")

    parser.add_argument('--val-preprocessed', action='store_true', default=True,
                        help="Directly use preprocessed data for validation dataloader.")

    args = parser.parse_args()

    # TODO: remove, used only for debug
    # ------------------------------------------------------------------------------------------------------------------
    # args.classification = True
    # args.loss_function = "focal"
    # args.model = 'transformer'
    # args.backbone = 'resnet50'
    # args.datapath = "data"
    # args.apple = True
    # args.train_preprocessed = True
    # args.val_preprocessed = True
    # args.MOTtrain = "MOT17"
    # args.MOTvalidation = "MOT17"
    # ------------------------------------------------------------------------------------------------------------------

    # There was no preconception of what to do  -cit.
    classification = args.classification

    # %% Set up parameters

    # Paths
    mot_path = args.datapath
    detections_file_folder = args.detection_folder
    detections_file_name = args.detection_file

    # MOT to use
    mot_train = args.MOTtrain
    mot_val = args.MOTvalidation

    # Preprocessed data?
    train_preprocessed = args.train_preprocessed
    val_preprocessed = args.val_preprocessed

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
    mps_fallback = args.apple_silicon  # Only if using MPS this should be true

    # Loss function
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
                raise Exception(
                    "Huber loss should not be used in a classification setting, please choose a different one")
        case 'bce' | 'mae' | 'mse':
            loss_function = IMPLEMENTED_LOSSES[loss_type]()
        case 'focal':
            loss_function = IMPLEMENTED_LOSSES[loss_type](alpha=alpha, gamma=gamma)
        # loss_not_initialized = True
        case 'berhu':
            raise NotImplemented("BerHu loss has not been implemented yet")
        case _:
            raise NotImplemented(
                "The chosen loss: " + loss_type +
                " has not been implemented yet. To see the available ones, run this script with the -h option")

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
                              classification=classification,
                              preprocessed=train_preprocessed,
                              feature_extraction_backbone=backbone)

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
                            mps_fallback=mps_fallback,
                            classification=classification,
                            preprocessed=val_preprocessed,
                            feature_extraction_backbone=backbone)

    network_dict = IMPLEMENTED_MODELS[args.model]

    model = Net(used_backbone=backbone, layer_size=l_size, n_target_edges=l_size, n_target_nodes=l_size, dtype=dtype,
                mps_fallback=mps_fallback, heads=heads, concat=False, dropout=args.dropout, add_self_loops=False,
                steps=messages, device=device, model_dict=network_dict,
                node_features_dim=ImgEncoder.output_dims[backbone], is_edge_model=not args.node_model,
                model_type=args.model)

    # %% Initialize the model

    model.train()

    optimizer = args.optimizer
    optimizer = AVAILABLE_OPTIMIZERS[optimizer](model.parameters(), lr=learning_rate)

    # Print information
    print("[INFO] Hyperparameters and info:")
    print("\n- Datasets:")
    print("\t- Dataset used for training: " + mot_train + " | validation: " + mot_val)
    print("\t- Subtrack length: " + str(subtrack_len))
    print("\t- Linkage window: " + str(linkage_window))
    print("\t- Sliding window: " + str(slide))
    print("\t- Setting: classification") if classification else print("\t- Setting: regression")
    print("\n- GNN Network:")
    print("\t- Backbone: " + backbone)
    print("\t- Number of heads: " + str(heads))
    print("\t- Number of message passing steps: " + str(messages))
    print("\t- Layer type: " + args.model)
    print("\t- Layer size: " + str(l_size))
    print("\t- Number of edge features: " + str(EDGE_FEATURES_DIM))
    print("\t- Dropout: " + str(args.dropout))
    print("\t- Prediction based on: Nodes" if args.node_model else "\t- Prediction based on: Edges")
    print("\n- Training:")
    print("\t- Loss function: " + loss_type)
    print("\t- Optimizer: " + args.optimizer)
    print("\t- Learning rate: " + str(learning_rate))
    print("")

    # %% Train the model

    train_loss, val_loss = train(model=model,
                                 train_loader=mot_train_dl,
                                 val_loader=mot_val_dl,
                                 loss_function=loss_function,
                                 optimizer=optimizer,
                                 epochs=epochs,
                                 device=device,
                                 mps_fallback=mps_fallback,
                                 alpha=alpha,
                                 gamma=gamma,
                                 reduction=reduction,
                                 classification=classification)

    print("\n[INFO] Final train loss: " + str(train_loss))

    print("\n[INFO] Final validation loss: " + str(val_loss))
