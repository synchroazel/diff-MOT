import argparse

from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from motclass import MotDataset
from utilities import *


# %% Function definitions
def test(model,
         val_loader,
         loss_function,
         device="cuda",
         alpha=.95,
         gamma=2,
         reduction='mean',
         one_threshold=0.5,
         zero_threshold=0.33,
         validation_mode=False):
    """
    Test the model using a given dataloader.
    Can be used for both testing and validation.
    """

    model = model.to(device)
    model.eval()

    val_loss, total_val_loss, total_0_1, total_1_0, total_1, total_0 = 0, 0, 0, 0, 0, 0
    avg_0_1, avg_1_0, avg_1, avg_0 = 0, 0, 0, 0

    # if not validation_mode:
    pbar_dl = tqdm(enumerate(val_loader), desc='[TQDM] Testing on track 1/? ', total=val_loader.n_subtracks)
   # else:
   #     pbar_dl = enumerate(val_loader)
    j = 0
    for _, data in pbar_dl:
        cur_track_idx = val_loader.cur_track + 1
        cur_track_name = val_loader.tracklist[val_loader.cur_track]

        with torch.no_grad():
            data = ToDevice(device.type)(data)

            cur_track_name = val_loader.tracklist[val_loader.cur_track]

            if validation_mode:
                if (cur_track_name not in MOT17_VALIDATION_TRACKS) and (cur_track_name not in MOT20_VALIDATION_TRACKS):
                    # Skip non-validation tracks if the function is called for validation
                    continue

            pred_edges = model(data)  # Get the predicted edge labels
            gt_edges = data.y  # Get the true edge labels
            val_loss = loss_function(pred_edges, gt_edges)

            zero_mask = pred_edges < zero_threshold
            one_mask = pred_edges > one_threshold

            pred_edges = torch.where(one_mask, 1., pred_edges)
            pred_edges = torch.where(zero_mask, 0, pred_edges)

            zero_mask = gt_edges < zero_threshold
            one_mask = gt_edges > one_threshold

            acc_ones = torch.where(pred_edges[one_mask] == 1.0, 1., 0.).mean().item()
            acc_zeros = torch.where(pred_edges[zero_mask] == 0., 1., 0.).mean().item()
            ones_as_zeros = torch.where(pred_edges[one_mask] == 0., 1., 0.).mean().item()
            zeros_as_ones = torch.where(pred_edges[zero_mask] == 1., 1., 0.).mean().item()

            total_val_loss += val_loss.item()
            total_0 += acc_zeros
            total_1 += acc_ones
            total_1_0 += ones_as_zeros
            total_0_1 += zeros_as_ones

            avg_0 = total_0 / (j + 1)
            avg_1 = total_1 / (j + 1)
            avg_0_1 = total_0_1 / (j + 1)
            avg_1_0 = total_1_0 / (j + 1)
            avg_val_loss = total_val_loss / (j + 1)

            j += 1

        avg_val_loss_msg = f'avg.Loss: {avg_val_loss:.4f} (last: {val_loss:.4f})'

        val_accs_msg = f"Accs: " \
                       f"[ 0 ✔ {avg_0:.2f} ] [ 1 ✔ {avg_1:.2f}] " \
                       f"[ 0 ✖ {avg_0_1 :.2f} ] [ 1 ✖ {avg_1_0 :.2f} ]"

        pbar_dl.set_description(
            f'[TQDM] Validationg on track {cur_track_idx}/{len(val_loader.tracklist)} ({cur_track_name}) | {avg_val_loss_msg} - {val_accs_msg}')

    if validation_mode:
        return avg_val_loss, avg_1 , avg_0 , avg_0_1 , avg_1_0



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='python test.py',
        description='Script for predicting outputs given a MOT dataset',
        epilog='Es: python test.py -D "datasets" -m "timeaware_500_resnet50-backbone.pkl" --MOT MOT20',
        formatter_class=argparse.RawTextHelpFormatter)

    # TODO: remove the default option before deployment
    parser.add_argument('-D', '--datapath', default="/media/dmmp/vid+backup/Data", type=str,
                        help="Dataset path."
                             "NB: This project assumes a MOT dataset, this project has been tested with MOT17 and MOT20.")

    parser.add_argument('--model_savepath', default="models_to_try",
                        help="Folder where models are loaded.")

    parser.add_argument('--output_savepath', default="saves/outputs",
                        help="Folder where outputs are saved.")

    parser.add_argument('-m', '--model', default="node-predictor_node-model-timeaware_edge-model-base_layer-size-500_backbone-resnet50_messages-6_trained_on__MOT20.pkl",
                        help="Name of the network."
                             "NB: the model must be stored in the specified save folder.")

    parser.add_argument('-M', '--MOT', default="MOT20",
                        help="MOT dataset of reference.")

    parser.add_argument('-T', '--split', default="train",
                        help="MOT dataset split.")

    parser.add_argument('--float16', action='store_true',
                        help="Whether to use half floats or not.")

    parser.add_argument('--apple', action='store_true',
                        help="Whether an apple device is in use with mps (required for some fallbacks).")

    parser.add_argument('-L', '--loss_function', default="huber",
                        help="Loss function to use."
                             "Implemented losses: huber, bce, focal, dice.")

    parser.add_argument('--reduction', default="mean",
                        help="Reduction logic for the loss."
                             "Implemented reductions: mean, sum.")

    # TODO: describe how it works
    parser.add_argument('-a', '--alpha', default=0.25, type=float,
                        help="Alpha parameter for the focal loss.")

    # TODO: describe how it works
    parser.add_argument('-g', '--gamma', default=2., type=float,
                        help="Gamma parameter for the focal loss.")

    # TODO: describe how it works
    parser.add_argument('-d', '--delta', default=.4, type=float,
                        help="Delta parameter for the huber loss.")

    # TODO: describe how it works
    parser.add_argument('--one_threshold', default=.5, type=float,
                        help="Threshold to transform a weight to 1.")

    # TODO: describe how it works
    parser.add_argument('--zero_threshold', default=.33, type=float,
                        help="Threshold to transform a weight to 0.")

    parser.add_argument('--detection_gt_folder', default="gt",
                        help="detection ground truth folder.")

    parser.add_argument('--detection_gt_file', default="gt.txt",
                        help="detection ground truth folder.")

    parser.add_argument('--subtrack_len', default=15, type=int,
                        help="Length of the subtrack NB: a value higher than 20 might require too much memory.")

    parser.add_argument('--linkage_window', default=5, type=int,
                        help="Linkage window for building the graph es: w 5 -> detections in frame 0 will connect to detections up to frame 5.")

    parser.add_argument('--slide', default=10, type=int,
                        help="Sliding window to adopt during testing. NB: suggested to be subtrack len - linkage window.")

    parser.add_argument('-k', '--knn', default=20, type=int,
                        help="K parameter for knn reduction."
                             "NB: a value lower than 20 may exclude ground truths. Set to 0 for no kNN.")

    parser.add_argument('--cosine', action='store_true',
                        help="Use cosine distance instead of euclidean distance.")

    parser.add_argument('--classification', action='store_true',
                        help="Work in classification setting instead of regression.")

    args = parser.parse_args()

    # %% Set up parameters

    one_threshold = args.one_threshold
    zero_threshold = args.zero_threshold

    # Paths
    mot_path = args.datapath
    saves_path = args.model_savepath
    output_path_folder = args.output_savepath

    # Model to load
    model_pkl = args.model

    # MOT to use
    mot = args.MOT

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
    mps_fallback = args.apple  # Only if using MPS this should be true

    # Loss function
    alpha = args.alpha
    delta = args.delta
    gamma = args.gamma
    reduction = args.reduction
    loss_type = args.loss_function
    loss_not_initialized = False
    match loss_type:
        case 'huber':
            loss_function = IMPLEMENTED_LOSSES[loss_type](delta=delta, reduction=reduction)
        case 'bce':
            loss_function = IMPLEMENTED_LOSSES[loss_type]()
        case 'focal':
            loss_function = IMPLEMENTED_LOSSES[loss_type](alpha=alpha, gamma=gamma, device=device)
        case 'berhu':
            raise NotImplemented("BerHu loss has not been implemented yet")
        case _:
            raise NotImplemented(
                "The chosen loss: " + loss_type + " has not been implemented yet. To see the available ones, run this script with the -h option")

    # Knn logic
    if args.knn <= 0:
        knn_args = None
    else:
        knn_args = {
            'k': args.knn,
            'cosine': args.cosine
        }

    classification = args.classification

    # %% Load the model

    model_path = os.path.normpath(os.path.join(saves_path, model_pkl))
    model = load_model_pkl(model_path, device=device).to(device)
    model.mps_fallback = mps_fallback
    model.eval()

    detections_file_folder = args.detection_gt_folder
    detections_file_name = args.detection_gt_file
    split = args.split

    # %% Set up the dataloader

    dataset_path = os.path.normpath(os.path.join(mot_path, mot))
    mot_train_dl = MotDataset(dataset_path=dataset_path,
                              split=split,
                              subtrack_len=subtrack_len,
                              slide=slide,
                              linkage_window=linkage_window,
                              detections_file_folder=detections_file_folder,
                              detections_file_name=detections_file_name,
                              device=device,
                              mps_fallback=mps_fallback,
                              dtype=dtype,
                              classification=classification)

    # %% Test the model

    test(model=model,
         val_loader=mot_train_dl,
         loss_function=loss_function,
         alpha=alpha,
         gamma=alpha,
         reduction=reduction,
         device=device,
         zero_threshold=zero_threshold,
         one_threshold=one_threshold)
