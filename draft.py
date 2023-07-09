import os

import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
import torch_geometric.data as pyg_data
import pickle
from motclass import *
from utilities import get_best_device

device = get_best_device()

mot20_path = "/media/dmmp/vid+backup/Data/MOT20/images"
# mot20_path = "/media/dmmp/vid+backup/Data/MOT17"
# mot20 = MotDataset(mot20_path, 'train', linkage_window=LINKAGE_TYPES["ALL"], device=device, det_resize=(32, 32))
# mot20 = MotDataset(mot20_path, 'train', linkage_window=5, device=device, det_resize=(32, 32))
# mot20 = MotDataset(mot20_path, 'train', linkage_window=15, device=device, det_resize=(32, 32))
mot20 = MotDataset(mot20_path, 'train', subtrack_len=-1, device=device, linkage_window=24, dtype=torch.float16, name="MOT20")

track = mot20[1]

graph = build_graph(**track.get_data(), device=device, dtype=torch.float16)

save_path = os.path.normpath(os.path.join("saves",*(str(track).split('/'))[0:-1]))
if not os.path.exists(save_path):
    os.makedirs(save_path)

file_name = os.path.normpath(os.path.join(save_path,str(track).split('/')[-1])) + ".pickle"
with open(file_name, 'wb') as f:
    pickle.dump(graph,f)
print("Graph saved as: " + file_name)
print(graph)

