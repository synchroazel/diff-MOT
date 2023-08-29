import warnings

from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from torchvision.ops import sigmoid_focal_loss

from utilities import *

warnings.filterwarnings("ignore")


def test(model,
         val_loader,
         loss_function,
         device,
         validation_mode=False):
    """
    Validate the model on a single subtrack, given a MOT dataloader and an index.
    """

    model = model.to(device)
    model.eval()

    val_loss, total_val_loss, total_0_1, total_1_0, total_1, total_0 = 0, 0, 0, 0, 0, 0
    avg_0_1, avg_1_0, avg_1, avg_0 = 0, 0, 0, 0

    pbar_dl = tqdm(enumerate(val_loader), desc='[TQDM] Testing on track 1/? ', total=val_loader.n_subtracks)

    j = 0
    for _, data in pbar_dl:
        cur_track_idx = val_loader.cur_track + 1

        with torch.no_grad():
            data = ToDevice(device.type)(data)

            cur_track_name = val_loader.tracklist[val_loader.cur_track]

            if validation_mode:
                if (cur_track_name not in MOT17_VALIDATION_TRACKS) and (cur_track_name not in MOT20_VALIDATION_TRACKS):
                    # Skip non-validation tracks if the function is called for validation
                    continue

            gt_edges = data.y

            # One-hot encoded y - INVERSE!
            oh_y = torch.nn.functional.one_hot(gt_edges.to(torch.int64), -1)

            # Edge attributes
            edge_attr = data.edge_attr

            # Edge indexes
            edge_index = data.edge_index

            _, pred_edges_oh = model.p_sample_loop(shape=(oh_y.shape[0], 2),
                                                   edge_feats=edge_attr,
                                                   node_feats=data.detections,
                                                   edge_index=edge_index)

            pred_edges = torch.where(pred_edges_oh[:, 1] > pred_edges_oh[:, 0], 0., 1.)

            val_loss = loss_function(data.y, pred_edges)

            zero_mask = gt_edges <= .5
            one_mask = gt_edges > .5

            acc_ones = torch.where(pred_edges[one_mask] == 1., 1., 0.).mean().item()
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
        return avg_val_loss, avg_1, avg_0, avg_0_1, avg_1_0
