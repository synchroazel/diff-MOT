import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
import torch_geometric.data as pyg_data

from motclass import *
from utilities import get_best_device

device = get_best_device()

mot20_path = "/media/dmmp/vid+backup/Data/MOT20/images"
mot20 = MotDataset(mot20_path, 'train', linkage_window=LINKAGE_TYPES["ALL"], device=device)
# mot20 = MotDataset(mot20_path, 'train', linkage_window=5, device=device)

track = mot20[0]

adj_list, node_feats, frame_times, edge_attr = track.get_data()

del track

graph = build_graph(adjacency_list=adj_list,
                    flattened_node=node_feats,
                    frame_times=frame_times,
                    edge_partial_attributes=edge_attr,
                    feature_extractor='resnet101',
                    device=device)


