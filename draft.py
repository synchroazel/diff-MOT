import pandas
import torch
import torchvision
from torch_geometric.transforms import ToDevice
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from model import Net
from motclass import MotDataset
from utilities import get_best_device
from utilities import load_model_pkl

device = get_best_device()

mot_train_dl = MotDataset(dataset_path='/media/dmmp/vid+backup/Data/MOT17',
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
                          classification=True)

# model = load_model_pkl("base_500_resnet50-backbone.pkl", device=device)  # regression
# model.mps_fallback = True

model = Net(backbone="ResNet50",
            layer_tipe="base",
            layer_size=500,
            dtype=torch.float32,
            edge_features_dim=6,
            heads=6,
            concat=False,
            dropout=0.3,
            add_self_loops=False,
            steps=6,
            device=device)

model.eval()

model = model.to(device)


nodes_dict = {}  # frame, bbox
id = 1
final_df = pandas.DataFrame()


def build_trajectory_rec(node_idx, pyg_graph, nx_graph, node_dists, nodes_todo, out, depth=0):

    global id
    global nodes_dict

    all_out_edges = nx_graph.out_edges(node_idx)

    # Remove edges going in the past
    all_out_edges = [item for item in all_out_edges if item[0] < item[1]]

    if len(all_out_edges) == 0:

        if depth == 0:  # it's an orphan node

            ### CHECK IF NEEDED ###

            orp_coords = pyg_graph.detections_coords[node_idx]
            orp_frame = pyg_graph.times[node_idx].tolist()[0]

            if (orp_frame, *orp_coords.tolist()) not in nodes_dict.keys():
                nodes_dict[(orp_frame, *orp_coords.tolist())] = id

            orp_id = nodes_dict[(orp_frame, *orp_coords.tolist())]

            out.append(
                {'frame': orp_frame,
                 'id': orp_id,
                 'bb_left': orp_coords[0].item(),
                 'bb_top': orp_coords[1].item(),
                 'bb_width': orp_coords[2].item(),
                 'bb_height': orp_coords[3].item(),
                 'conf': -1,
                 'x': -1,
                 'y': -1,
                 'z': -1}
            )

            ### CHECK IF NEEDED ###

        return

    # Find best edge to keep
    # TODO: find best peso
    best_edge_idx = torch.tensor([node_dists[n1, n2] for n1, n2 in all_out_edges]).argmin()
    best_edge = list(all_out_edges)[best_edge_idx]

    # Remove all other edges
    nx_graph.remove_edges_from(
        [item for item in list(all_out_edges) if item != list(all_out_edges)[best_edge_idx]]
    )

    n1, n2 = best_edge

    # Remove nodes from to-visit set if there
    if n1 in nodes_todo: nodes_todo.remove(n1)
    if n2 in nodes_todo: nodes_todo.remove(n2)
    # TODO: make more light

    n1_coords = torchvision.ops.box_convert(pyg_graph.detections_coords[n1], 'xyxy','xywh').tolist()
    n2_coords = torchvision.ops.box_convert(pyg_graph.detections_coords[n2], 'xyxy','xywh').tolist()

    n1_frame = pyg_graph.times[n1].tolist()[0]
    n2_frame = pyg_graph.times[n2].tolist()[0]

    # n1_coords = box_convert(n1_coords, in_fmt='xyxy', out_fmt='xywh')
    # n2_coords = box_convert(n2_coords, in_fmt='xyxy', out_fmt='xywh')

    if (n1_frame, *n1_coords.tolist()) not in nodes_dict.keys():
        nodes_dict[(n1_frame, *n1_coords.tolist())] = id

    if (n2_frame, *n2_coords.tolist()) not in nodes_dict.keys():
        nodes_dict[(n2_frame, *n2_coords.tolist())] = id

    n1_id = nodes_dict[(n1_frame, *n1_coords.tolist())]

    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    out.append(
        {'frame': n1_frame + 1,
         'id': n1_id,
         'bb_left': n1_coords[0].item(),
         'bb_top': n1_coords[1].item(),
         'bb_width': n1_coords[2].item(),
         'bb_height': n1_coords[3].item(),
         'conf': -1,
         'x': -1,
         'y': -1,
         'z': -1}
    )

    # print(
    #     n1_frame, n1_id, n1_coords[0].item(), n1_coords[1].item(), n1_coords[2].item(), n1_coords[3].item(),
    #     -1, -1, -1, -1
    # )

    depth += 1

    build_trajectory_rec(node_idx=n2, pyg_graph=pyg_graph, nx_graph=nx_graph, node_dists=node_dists, nodes_todo=nodes_todo,
                         out=out, depth=depth)


def build_trajectories(graph, preds, ths=.33):
    global nodes_dict
    global id
    pyg_graph = graph.clone().detach()

    mask = torch.where(preds > float(ths), True, False)

    masked_preds = preds[mask]  # Only for regression

    pred_edges = pyg_graph.edge_index.t()[mask]

    out = []

    node_dists = torch.cdist(pyg_graph.pos, pyg_graph.pos, p=2)

    nodes_todo = list(range(pyg_graph.num_nodes))  # Used as a stack
    nodes_todo.reverse()
    # Create a NetwrokX graph
    pyg_graph.edge_index = pred_edges.t()
    nx_graph = to_networkx(pyg_graph)

    while len(nodes_todo) > 0:
        build_trajectory_rec(node_idx=nodes_todo.pop(), pyg_graph=pyg_graph, nx_graph=nx_graph, node_dists=node_dists,
                             nodes_todo=nodes_todo, out=out)
        id += 1
        # print(f"\rRemaining nodes to visit: {len(nodes_todo)}     ", end="")

    return out

for _, data in tqdm(enumerate(mot_train_dl), desc='[TQDM] Converting tracklet', total=mot_train_dl.n_subtracks):
    data = ToDevice(device.type)(data)
    # preds = model(data)
    # out = build_trajectories(data, preds=preds)
    out = build_trajectories(data, data.y)

    df = pandas.DataFrame.from_dict(out)

    final_df = pandas.concat([final_df, df], ignore_index=True)

final_df.sort_values(by=['id', 'frame']).to_csv("test.csv")


