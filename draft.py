import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
import torch_geometric.data as pyg_data

from motclass import *
from utilities import get_best_device

device = get_best_device()

# mot20_path = '/media/dmmp/vid+backup/Data/MOT20/images'
mot20_path = '/media/dmmp/vid+backup/Data/MOT17'
mot20 = MotDataset(mot20_path, 'train', linkage_type="ADJACENT", device=device, det_resize=(256,256))

track = mot20[0]

adjacency_list, flattened_node_features, frame_times, edge_attr = track.get_data(limit=-1)

del track
graph = build_graph(adjacency_list=adjacency_list, flattened_node=flattened_node_features, frame_times=frame_times,
                    edge_partial_attributes=edge_attr, feature_extractor='resnet101', device='cuda')


