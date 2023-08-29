import io
import json
import logging
import os
import pickle
from tqdm import tqdm

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import HuberLoss, BCEWithLogitsLoss, MSELoss, L1Loss
from torchgeometry.losses import DiceLoss
from torch_geometric.utils import to_networkx
import torch.nn.functional as F


def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_best_device():
    """Function used to get the best device between cuda, mps and cpu"""
    if torch.cuda.is_available():
        print("[INFO] Using CUDA.")
        return torch.device("cuda")
    elif torch.has_mps:
        print("[INFO] Using MPS.")
        return torch.device("mps")
    else:
        print("[INFO] No GPU found. Using CPU.")
        return torch.device("cpu")


def minkowski_distance(a: list, b: list, power=2):
    """
    This function calculates the Minkowski distance between two data points
    :param a: first data point
    :param b: second data point
    :param power: type of distance (1: Manhattan, 2: Euclidean, 3: Cosine)
    """
    if power < 1:
        raise Exception("Power of Minkowski distance should not be less than 1")
    if a is None or b is None:
        raise Exception("While calculating the distance, one of the two elements was none")
    # return np.float16(sum(abs(e1 - e2) ** power for e1, e2 in zip(a, b)) ** (1 / power))
    return sum(abs(e1 - e2) ** power for e1, e2 in zip(a, b)) ** (1 / power)


def draw_graph(graph, track):
    G = to_networkx(graph, to_undirected=True)

    n_sum = [0] + list(np.cumsum(track.n_nodes))

    for i, n in enumerate(track.n_nodes):
        for j in range(n):
            G.nodes[j + n_sum[i]]['level'] = i

    pos = nx.multipartite_layout(G, subset_key="level")

    nx.draw(G, pos=pos, with_labels=True)


def save_graph(graph, track):
    save_path = os.path.normpath(os.path.join("saves", *(str(track).split('/'))[0:-1]))

    create_folders(save_path)

    file_name = os.path.normpath(os.path.join(save_path, str(track).split('/')[-1])) + ".pickle"

    with open(file_name, 'wb') as f:
        pickle.dump(graph.cpu(), f)

    print("[INFO] Graph saved as " + file_name)


def load_graph(pickle_path):
    with open(pickle_path, 'rb') as f:
        graph = pickle.load(f)

    print("[INFO] Graph loaded from " + pickle_path)

    return graph


def save_model(model: torch.nn.Module,
               savepath_adds: dict = None,
               savepath: str = "saves/models",
               mode: str = "pkl",
               mps_fallback: bool = False,
               classification: bool = False,
               epoch: int = 0,
               epoch_info: dict = None,
               node_model_name='timeaware',
               edge_model_name='base'):
    if mps_fallback:
        model.to(torch.device('cpu'))

    # path components
    technique = "classification" if classification else "regression"
    epoch = "Epoch_" + str(epoch)

    if node_model_name is not None and edge_model_name is not None:
        model_name = node_model_name + "_" + edge_model_name
    else:
        model_name = "model"

    savepath = os.path.normpath(
        os.path.join(
            savepath,
            technique,
            model_name,
            epoch
        )
    )
    create_folders(savepath)
    savepath = os.path.normpath(
        os.path.join(
            savepath, str(model)
        )
    )

    match mode:

        case "pkl":

            if savepath_adds:
                for key in savepath_adds:
                    savepath += "_" + str(key) + "_" + str(savepath_adds[key])

            pkl_savepath = savepath + ".pkl"

            pickle.dump(model, open(pkl_savepath, "wb"))

            if epoch_info:
                json_savepath = savepath + ".json"
                js = json.dumps(epoch_info, sort_keys=True, indent=4, separators=(',', ': '))
                with open(json_savepath, 'w+') as f:
                    f.write(js)

        case "weights":
            pass  # TODO implement

        case _:
            logging.error("Saving mode not implemented.")

    if mps_fallback:
        model.to(torch.device('mps'))


def custom_load_pkl(pkl_path, device='cpu'):
    class CustomUnpickler(pickle.Unpickler):  # this is necessary to deal with MPS
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=device)
            else:
                return super().find_class(module, name)

    with open(pkl_path, 'rb') as file:
        unpickler = CustomUnpickler(file)
        obj = unpickler.load()
    return obj


class SigmoidFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2, device='cuda'):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def shuffle_tensor(tensor):
    idx = torch.randperm(tensor.shape[0])
    t = tensor[idx].view(tensor.size())
    return t


def check_sanity(dataloader):
    for _ in tqdm(dataloader, desc="[TQDM] Verifying dataloader sanity", leave=True):
        pass


# TODO: add more?
AVAILABLE_OPTIMIZERS = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'Rprop': torch.optim.Rprop,
    'SGD': torch.optim.SGD,
    'RMSprop': torch.optim.RMSprop,
    'Adagrad': torch.optim.Adagrad,
    'Adamax': torch.optim.Adamax,
    'Adadelta': torch.optim.Adadelta,

}

# implemented losses
IMPLEMENTED_LOSSES = {
    'huber': HuberLoss,
    'bce': BCEWithLogitsLoss,
    'focal': SigmoidFocalLoss,
    'mae': L1Loss,
    'mse': MSELoss,
    'dice': DiceLoss,
    'berhu': None,
}

# %% Config stuff

EDGE_FEATURES_DIM = 6
LINKAGE_TYPE_ALL = -1
LINKAGE_TYPE_ADJACENT = 0
EPSILON = 0.0001  # Used to avoid infinity

MOT20_VALIDATION_TRACKS = {'MOT20-01': True, 'MOT20-02': True}  # dictionary for having a faster check access
MOT17_VALIDATION_TRACKS = {'MOT17-02-SDP': True,
                           'MOT17-02-FRCNN': True,
                           'MOT17-02-DPM': True,
                           'MOT17-09-DPM': True,
                           'MOT17-09-SDP': True,
                           'MOT17-09-FRCNN': True,
                           'MOT17-13-DPM': False,
                           'MOT17-13-FRCNN': False,
                           'MOT17-13-SDP': False,
                           }
