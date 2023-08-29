# %%
import torch
import pickle

from tqdm import tqdm

from diff_motclass import MotDataset
from puzzle_diff.model.spatial_diffusion import *
from utilities import get_best_device
from torch_geometric.transforms import ToDevice
from utilities import check_sanity

device = torch.device('cpu')  # get_best_device()

# %%

mot_train_dl = MotDataset(dataset_path='data/MOT17',
                          split='train',
                          subtrack_len=15,
                          slide=10,
                          linkage_window=5,
                          detections_file_folder='gt',
                          detections_file_name='gt.txt',
                          dl_mode=True,
                          device=device,
                          dtype=torch.float32,
                          preprocessed=True,
                          mps_fallback=True,
                          classification=True,
                          feature_extraction_backbone='efficientnet_v2_l')

# %%

def validation_step():

    avg_val_loss = 0

    pbar = tqdm(mot_train_dl, total=mot_train_dl.n_subtracks)

    i = 1

    total_val_loss, total_0, total_1, total_1_0, total_0_1 = 0, 0, 0, 0, 0

    for data in pbar:

        if i == lim:
            break

        data = ToDevice(device.type)(data)

        oh_y = torch.nn.functional.one_hot(data.y.to(torch.int64), -1)
        edge_attr = data.edge_attr.to(device)
        edge_index = data.edge_index.to(device)


        _, pred_edges_oh = diff_model.p_sample_loop(shape=(oh_y.shape[0], 2),
                                                    edge_feats=edge_attr,
                                                    node_feats=data.detections,
                                                    edge_index=edge_index)

        pred_edges = torch.where(pred_edges_oh[:, 1] > pred_edges_oh[:, 0], 1., 0.)

        val_loss = F.smooth_l1_loss(data.y, pred_edges)

        avg_val_loss = (val_loss.item() + avg_val_loss) / i

        pbar.set_description(f"Validation - Avg. Val Loss: {avg_val_loss:.6f}")

        i += 1
        pbar.update(1)
        del data

def training_step():
    avg_loss = 0

    pbar = tqdm(mot_train_dl, total=mot_train_dl.n_subtracks)

    i = 1

    for data in pbar:

        if i == lim:
            break

        data = ToDevice(device.type)(data)

        oh_y = torch.nn.functional.one_hot(data.y.to(torch.int64), -1)
        time = torch.zeros((oh_y.shape[0])).long()
        edge_attr = data.edge_attr.to(device)
        edge_index = data.edge_index.to(device)

        loss = diff_model.p_losses(
            x_start=oh_y,
            t=time,
            loss_type="huber",
            node_feats=data.detections,
            edge_index=edge_index,
            edge_feats=edge_attr
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = (loss.item() + avg_loss) / i

        pbar.set_description(f"Epoch {e} - Avg. Loss: {avg_loss:.6f}")

        i += 1
        pbar.update(1)
        del data

def save_model(model, name):
    f = open(name, 'wb')
    pickle.dump(model,f)
    f.close()

# %%

steps = 300
epochs = 20
lim = 150

diff_model = GNN_Diffusion(steps=steps).to(device)
optimizer = Adafactor(diff_model.parameters(), lr=0.001, relative_step=False)

# %%

# save_model(diff_model, 'modello_stronzo.pkl')

# for e in range(epochs):
#
#     validation_step()
#
#     print("---")
#
#     training_step()
#
# print('Done with training.')

# %%

for e in range(epochs):

    pbar = tqdm(enumerate(mot_train_dl), total=mot_train_dl.n_subtracks)

    for i, data in pbar:

        data = ToDevice(device.type)(data)

        oh_y = torch.nn.functional.one_hot(data.y.to(torch.int64), -1)
        time = torch.zeros((oh_y.shape[0])).long()
        edge_attr = data.edge_attr.to(device)
        edge_index = data.edge_index.to(device)

        loss = diff_model.p_losses(
            x_start=oh_y,
            t=time,
            loss_type="huber",
            node_feats=data.detections,
            edge_index=edge_index,
            edge_feats=edge_attr
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)

        # -- #

        if i < lim:
            pbar.set_description(f"Epoch {e} - Loss: {loss.item():.6f}")
            continue

        data_v = mot_train_dl[i+1]

        data_v = ToDevice(device.type)(data_v)

        oh_y = torch.nn.functional.one_hot(data_v.y.to(torch.int64), -1)
        edge_attr = data_v.edge_attr.to(device)
        edge_index = data_v.edge_index.to(device)

        _, pred_edges_oh = diff_model.p_sample_loop(shape=(oh_y.shape[0], 2),
                                                    edge_feats=edge_attr,
                                                    node_feats=data_v.detections,
                                                    edge_index=edge_index)

        pred_edges = torch.where(pred_edges_oh[:, 1] > pred_edges_oh[:, 0], 0., 1.)

        val_loss = F.smooth_l1_loss(data_v.y, pred_edges)

        val_acc = torch.where(pred_edges == data_v.y, 1., 0.).mean().item()

        pbar.set_description(f"Epoch {e} - Loss: {loss.item():.6f} - Val. Loss: {val_loss.item():.6f} - Acc. on next: {val_acc:.6f}")


















# # save_model(diff_model, 'modello_bello.pkl')
#
# # unpickle models
# diff_model_bello = pickle.load(open('modello_bello.pkl', 'rb'))
# diff_model_stronzo = pickle.load(open('modello_stronzo.pkl', 'rb'))
#
#
# for data in mot_train_dl:
#
#     data = ToDevice(device.type)(data)
#
#     gt_edges = data.y
#     oh_y = torch.nn.functional.one_hot(data.y.to(torch.int64), -1)
#     time = torch.zeros((oh_y.shape[0])).long()
#     edge_attr = data.edge_attr.to(device)
#     edge_index = data.edge_index.to(device)
#
#
#     _, pred_edges_oh = diff_model_bello.p_sample_loop(shape=(oh_y.shape[0], 2),
#                                                 edge_feats=edge_attr,
#                                                 node_feats=data.detections,
#                                                 edge_index=edge_index)
#
#     pred_edges_bello = torch.where(pred_edges_oh[:, 1] > pred_edges_oh[:, 0], 0., 1.)
#
#     _, pred_edges_oh = diff_model_stronzo.p_sample_loop(shape=(oh_y.shape[0], 2),
#                                                       edge_feats=edge_attr,
#                                                       node_feats=data.detections,
#                                                       edge_index=edge_index)
#
#     pred_edges_stronzo = torch.where(pred_edges_oh[:, 1] > pred_edges_oh[:, 0], 0., 1.)
#
#     break
#
# print("Modello bello acc: ",
#       (torch.where(pred_edges_bello == gt_edges, 1., 0.).sum() / gt_edges.shape[0]).item() * 100)
#
# print("Modello stronzo acc: ",
#       (torch.where(pred_edges_stronzo == gt_edges, 1., 0.).sum() / gt_edges.shape[0]).item() * 100)