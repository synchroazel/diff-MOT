# %%
import torch
import pickle

from motclass import MotDataset
from puzzle_diff.model.spatial_diffusion import *
from utilities import get_best_device
from torch_geometric.transforms import ToDevice

device = torch.device('cpu')  # get_best_device()

# %%
mot_train_dl = MotDataset(dataset_path='data/MOT17',
                          split='train',
                          subtrack_len=15,
                          slide=15,
                          linkage_window=5,
                          detections_file_folder='gt',
                          detections_file_name='gt.txt',
                          dl_mode=True,
                          knn_pruning_args={'k': 20, 'cosine': False},
                          device=device,
                          dtype=torch.float32,
                          classification=True,
                          mps_fallback=True)

# data = mot_train_dl[0]
# data = ToDevice(device.type)(data)
#
# # save to pickle
# with open('data.pkl', 'wb') as f:
#     pickle.dump(data, f)

# load object

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

data = ToDevice(device.type)(data)

# %%
diff_model = GNN_Diffusion().to(device)

# %%
# One-hot encoded y
oh_y = torch.where(torch.vstack((data.y, data.y)).t().cpu() == torch.tensor([1., 1.]),
                   torch.tensor([1., 0]),
                   torch.tensor([0., 1.])).to(device)

# Diffusion times
# times = torch.zeros((oh_y.shape[0])).to(device).int()
time = torch.zeros((oh_y.shape[0])).long()

# Edge attributes
edge_attr = data.edge_attr.to(device)

# Edge indexes
edge_index = data.edge_index.to(device)

# %%

# diff_model.training_step(oh_y, edge_attr, edge_index,  0, data.y)

# diff_model.q_sample(oh_y, t)

# loss = diff_model.p_losses(
#     oh_y,
#     time,
#     loss_type="huber",
#     cond=edge_attr,
#     edge_index=edge_index,
#     edge_attr=edge_attr
# )

# out = diff_model(xy_pos=oh_y, time=time, patch_rgb=edge_index, edge_index=edge_attr)

# _, denoised = diff_model.p_sample_loop(shape=(oh_y.shape[0], 2), cond=edge_attr, edge_index=edge_index)

# diff_model.p_sample(oh_y, time, 0, cond=edge_attr, edge_index=edge_index, sampling_func=diff_model.p_sample_ddpm)