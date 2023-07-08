import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
import torch_geometric.data as pyg_data

from motclass import *
from utilities import get_best_device

device = get_best_device()

# mot20_path = "/media/dmmp/vid+backup/Data/MOT20/images"
mot20_path = "/media/dmmp/vid+backup/Data/MOT17"
# mot20 = MotDataset(mot20_path, 'train', linkage_window=LINKAGE_TYPES["ALL"], device=device, det_resize=(32, 32))
# mot20 = MotDataset(mot20_path, 'train', linkage_window=5, device=device, det_resize=(32, 32))
# mot20 = MotDataset(mot20_path, 'train', linkage_window=15, device=device, det_resize=(32, 32))
mot20 = MotDataset(mot20_path, 'train', subtrack_len=-1, device=device)

track = mot20[1]

graph = build_graph(**track.get_data(), device=device)



