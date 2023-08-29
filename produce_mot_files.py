import argparse
from collections import OrderedDict

import pandas
import torchvision
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from motclass import MotDataset
from utilities import *
from utilities import get_best_device
from utilities import custom_load_pkl

device = get_best_device()

parser = argparse.ArgumentParser()

parser.add_argument('-D', '--datapath', default="data",
                    help="path to the folder containing the datasets.")

parser.add_argument('-o', '--outpath', default="trackers",
                    help="output path.")

parser.add_argument('-e', '--experiment',
                    help="experiment folder name.")

parser.add_argument('-m', '--model',
                    help="path to the model to load.")

parser.add_argument('-b', '--backbone', default="efficientnet_v2_l",
                    help="visual backbone to use in the model.")

parser.add_argument('-r', '--regression', action='store_true')

parser.add_argument('-v', '--validation_only', action='store_true')

parser.add_argument('--mot', default='MOT17')

parser.add_argument('--detection_folder', default="gt",
                    help="detection ground truth folder.")

parser.add_argument('--detection_file', default="gt.txt",
                    help="detection ground truth folder.")

parser.add_argument('--diff-steps', type=int,
                    help="number of steps for the diff.")

args = parser.parse_args()

datapath = args.datapath
outpath = os.path.join(args.outpath, args.experiment)
model = custom_load_pkl(args.model, device=device)
classification = not args.regression
validation = args.validation_only
mot = args.mot
backbone = args.backbone

if args.diff_steps is not None:
    model.steps = args.diff_steps

folder = args.detection_folder
file = args.detection_file

data_loader = MotDataset(dataset_path=os.path.join(datapath, mot),
                         split='train',
                         subtrack_len=15,
                         slide=10,
                         linkage_window=5,
                         detections_file_folder=folder,
                         detections_file_name=file,
                         dl_mode=True,
                         device=device,
                         dtype=torch.float32,
                         classification=classification,
                         feature_extraction_backbone=backbone)

nodes_dict = {}
id, track_frame = 1, 1

final_df = pandas.DataFrame(columns=['frame',
                                     'id',
                                     'bb_left',
                                     'bb_top',
                                     'bb_width',
                                     'bb_height',
                                     'conf',
                                     'x',
                                     'y',
                                     'z'])


def build_trajectory_rec(node_idx: int, pyg_graph, nx_graph, node_dists, nodes_todo: OrderedDict,
                         depth: int = 0) -> bool:
    global id
    global nodes_dict
    global final_df
    global track_frame

    n1_frame = track_frame + pyg_graph.times[node_idx].tolist()[0]
    n1_coords = torchvision.ops.box_convert(pyg_graph.detections_coords[node_idx], 'xyxy', 'xywh').tolist()
    c1 = (n1_frame, *n1_coords) in nodes_dict.keys()
    if c1 and nodes_dict[(n1_frame, *n1_coords)][1]:
        return False

    new_id = True
    all_out_edges = nx_graph.out_edges(node_idx)

    # Remove edges going in the past
    all_out_edges = [item for item in all_out_edges if ((item[0] < item[1]) and (item[1] in nodes_todo))]

    if len(all_out_edges) == 0:

        if depth == 0:  # it's an orphan node

            ### CHECK IF NEEDED ###

            orp_coords = torchvision.ops.box_convert(pyg_graph.detections_coords[node_idx], 'xyxy', 'xywh').tolist()
            orp_frame = track_frame + pyg_graph.times[node_idx].tolist()[0]

            if (orp_frame, *orp_coords) not in nodes_dict.keys():
                nodes_dict[(orp_frame, *orp_coords)] = [id, False]

            orp_id, _ = nodes_dict[(orp_frame, *orp_coords)]
            new_row = {'frame': orp_frame,
                       'id': orp_id,
                       'bb_left': orp_coords[0],
                       'bb_top': orp_coords[1],
                       'bb_width': orp_coords[2],
                       'bb_height': orp_coords[3],
                       'conf': -1,
                       'x': -1,
                       'y': -1,
                       'z': -1}
            final_df.loc[len(final_df)] = new_row

            ### CHECK IF NEEDED ###

        return new_id  # True

    # Find best edge to keep
    best_edge_idx = torch.tensor([nx_graph[n1][n2]['edge_weights'] for n1, n2 in all_out_edges]).argmax()
    best_edge = list(all_out_edges)[best_edge_idx]

    # Remove all other edges
    nx_graph.remove_edges_from(
        [item for item in list(all_out_edges) if item != list(all_out_edges)[best_edge_idx]]
    )

    n1, n2 = best_edge

    # Remove nodes from to-visit set if there
    # node 1 is popped at the start and deleted in the previous iteration
    del nodes_todo[n2]

    n2_coords = torchvision.ops.box_convert(pyg_graph.detections_coords[n2], 'xyxy', 'xywh').tolist()

    n2_frame = track_frame + pyg_graph.times[n2].tolist()[0]

    c2 = (n2_frame, *n2_coords) in nodes_dict.keys()

    if not c1:
        nodes_dict[(n1_frame, *n1_coords)] = [id, True]
    else:
        new_id = False
        nodes_dict[(n1_frame, *n1_coords)][1] = True
    node_id, _ = nodes_dict[(n1_frame, *n1_coords)]
    if not c2:
        nodes_dict[(n2_frame, *n2_coords)] = [node_id, False]

    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    if not c1:
        final_df.loc[len(final_df)] = {'frame': n1_frame,
                                       'id': node_id,
                                       'bb_left': n1_coords[0],
                                       'bb_top': n1_coords[1],
                                       'bb_width': n1_coords[2],
                                       'bb_height': n1_coords[3],
                                       'conf': -1,
                                       'x': -1,
                                       'y': -1,
                                       'z': -1}
    if not c2:  # c1 will always be true after the first iteration, because we also add the second node
        final_df.loc[len(final_df)] = {'frame': n2_frame,
                                       'id': node_id,
                                       'bb_left': n2_coords[0],
                                       'bb_top': n2_coords[1],
                                       'bb_width': n2_coords[2],
                                       'bb_height': n2_coords[3],
                                       'conf': -1,
                                       'x': -1,
                                       'y': -1,
                                       'z': -1}

    depth += 1

    _ = build_trajectory_rec(node_idx=n2, pyg_graph=pyg_graph, nx_graph=nx_graph, node_dists=node_dists,
                             nodes_todo=nodes_todo,
                             depth=depth)
    return new_id


create_folders(outpath)


def build_trajectories(graph, preds, ths=.33, classification=False):
    global nodes_dict
    global id
    pyg_graph = graph.clone().detach()

    if classification:
        preds = torch.sigmoid(preds)

    mask = torch.where(preds > float(ths), True, False)

    masked_preds = preds[mask]  # Only for regression
    pred_edges = pyg_graph.edge_index.t()[mask]

    node_dists = torch.cdist(pyg_graph.pos, pyg_graph.pos, p=2)

    nodes_todo = OrderedDict([(node, node) for node in range(0, pyg_graph.num_nodes)])

    # Create a NetwrokX graph
    pyg_graph.edge_index = pred_edges.t()
    pyg_graph.edge_weights = masked_preds

    nx_graph = to_networkx(pyg_graph, edge_attrs=['edge_weights'])

    while len(nodes_todo) > 0:
        starting_point = nodes_todo.popitem(last=False)[0]
        new_id = build_trajectory_rec(node_idx=starting_point, pyg_graph=pyg_graph, nx_graph=nx_graph,
                                      node_dists=node_dists,
                                      nodes_todo=nodes_todo)
        if new_id:
            id += 1


previous_track_idx = 0
validation_only = validation

for _, data in tqdm(enumerate(data_loader), desc='[TQDM] Converting tracklet',
                    total=data_loader.n_subtracks):  # todo: explain track
    cur_track_idx = data_loader.cur_track
    cur_track_name = data_loader.tracklist[data_loader.cur_track]

    if cur_track_idx != previous_track_idx and cur_track_idx != 0:
        cur_track_name = data_loader.tracklist[previous_track_idx]
        final_df.sort_values(by=['id', 'frame']).drop_duplicates().to_csv(outpath + "/" + cur_track_name + ".txt",
                                                                          index=False, header=False)
        previous_track_idx += 1
        # Reset values
        nodes_dict = {}  # frame, bbox
        id = 1
        track_frame = 1
        final_df = pandas.DataFrame(columns=['frame',
                                             'id',
                                             'bb_left',
                                             'bb_top',
                                             'bb_width',
                                             'bb_height',
                                             'conf',
                                             'x',
                                             'y',
                                             'z'])

    if validation_only and (
            (cur_track_name not in MOT17_VALIDATION_TRACKS) and (cur_track_name not in MOT20_VALIDATION_TRACKS)):
        continue

    data = ToDevice(device.type)(data)

    gt_edges = data.y

    oh_y = torch.nn.functional.one_hot(gt_edges.to(torch.int64), -1)
    edge_attr = data.edge_attr
    edge_index = data.edge_index

    with torch.no_grad():
        _, pred_edges_oh = model.p_sample_loop(shape=(oh_y.shape[0], 2),
                                               edge_feats=edge_attr,
                                               node_feats=data.detections,
                                               edge_index=edge_index)

    preds = torch.where(pred_edges_oh[:, 1] > pred_edges_oh[:, 0], 1., 0.)

    build_trajectories(data, preds=preds, classification=classification)

    track_frame += data_loader.slide

cur_track_name = data_loader.tracklist[cur_track_idx]

final_df.sort_values(by=['id', 'frame']).drop_duplicates().to_csv(outpath + "/" + cur_track_name + ".txt",
                                                                  index=False,
                                                                  header=False)
