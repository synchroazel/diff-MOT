import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
import torch_geometric.data as pyg_data

from motclass import MotDataset
from utilities import get_best_device

device = get_best_device()

mot20_path = '/media/dmmp/vid+backup/Data/MOT20/images'
mot20 = MotDataset(mot20_path, 'train', linkage_type="ADJACENT", device=device)

track = mot20[0]

adjacency_matrix, flattened_node_features, frame_times, edge_attr = track.get_data(limit=-1)

graph = pyg_data.Data(
    detections=flattened_node_features,
    num_nodes=flattened_node_features.shape[0],
    times=frame_times,
    edge_index=adjacency_matrix,
    edge_attr=edge_attr,
)

