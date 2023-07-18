import logging
import os
import pickle

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx


def create_subtracks(track_directory: str, frames: int) -> None:
    """
    This function is used to create subtracks of a MOT track.
    To avoid memory cluttering, it creates symbolic links in the subfolders.
    """

    images = os.listdir(track_directory)
    images = [x for x in images if x[-4] == '.']

    subtrack_base = os.path.normpath(os.path.join(track_directory, "subtrack_"))

    j = 0

    for i, image in enumerate(images):

        image_path = os.path.normpath(os.path.join(track_directory, image))

        if i % frames == 0:
            # create subfolder
            subfolder = subtrack_base + str(j)
            j += 1
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

        # create symbolic link
        try:
            os.symlink(src=image_path, dst=os.path.normpath(os.path.join(subfolder, image)))
        except FileExistsError:
            pass


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

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = os.path.normpath(os.path.join(save_path, str(track).split('/')[-1])) + ".pickle"

    with open(file_name, 'wb') as f:
        pickle.dump(graph.cpu(), f)

    print("[INFO] Graph saved as " + file_name)


def load_graph(pickle_path):
    with open(pickle_path, 'rb') as f:
        graph = pickle.load(f)

    print("[INFO] Graph loaded from " + pickle_path)

    return graph


def save_model(model, savepath="saves/models", mode="pkl", mps_fallback=False):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    if mps_fallback:
        model.to(torch.device('cpu'))

    match (mode):

        case "pkl":
            model_savepath = os.path.normpath(os.path.join(savepath, str(model) + ".pkl"))
            pickle.dump(model, open(model_savepath, "wb"))

        case "weights":
            pass  # TODO implement

        case _:
            logging.error("Saving mode not implemented.")

    if mps_fallback:
        model.to(torch.device('mps'))
