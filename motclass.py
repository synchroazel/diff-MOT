"""Set of classes used to deal with datasets and tracks"""
import os

import numpy as np
import torch
import torch_geometric.data as pyg_data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import box_convert
from tqdm import tqdm
from encoders import ImgEncoder

from utilities import minkowski_distance

LINKAGE_TYPES = {
    "ADJACENT": 0,
    "ALL": -1
}


def build_graph(adjacency_list: torch.Tensor,
                flattened_node: torch.Tensor,
                frame_times: torch.Tensor,
                edge_partial_attributes: torch.Tensor,
                feature_extractor: str,
                weights="DEFAULT",
                device='cuda') -> pyg_data.Data:
    """
    This function's purpose is to process the output of `track.get_data()` to build an appropriate graph.

    :param adjacency_list: tensor which represents the adjacency list. This will be used as 'edge_index'
    :param flattened_node: tensor which represents the node image
    :param frame_times: time distances of the frames
    :param edge_partial_attributes: at the moment, get_data returns just the center distance
    :param feature_extractor: feature extractor to use for calculating feature distances
    :param weights: weights to use for the feature extractor
    :param device: device to use, either mps, cuda or cpu

    :return: a Pytorch Geometric graph object
    """

    feature_extractor_net = ImgEncoder(feature_extractor, weights=weights, device=device)

    flattened_node = flattened_node.float()
    number_of_nodes = len(flattened_node)

    # NODES processing - image features extraction (using batches)

    node_features = None
    batch_size = 8

    pbar = tqdm(range(0, number_of_nodes, batch_size), desc="Processing nodes")

    for i in pbar:

        if i + batch_size > number_of_nodes:
            batch_tensor = flattened_node[i:, :, :, :]
        else:
            batch_tensor = flattened_node[i:i + batch_size, :, :, :]
        with torch.no_grad():
            f = feature_extractor_net(batch_tensor)
        if node_features is None:
            node_features = torch.zeros(f.shape[1]).to(device)  # otherwise we cannot stack

        node_features = torch.vstack((node_features, f))

    # f = f.reshape((batch_size, f.shape[1] * f.shape[2] * f.shape[3]))

    node_features = node_features[1:, :]

    # EDGES processing - edge features computation

    # pre-compute the # of batches
    n_batches = int(np.ceil(adjacency_list.shape[0] / batch_size))

    pbar = tqdm(range(0, int(n_batches)), desc="Processing edges")

    # precompute edge features, vstack causes memory leak
    final_edge_attributes = torch.zeros((edge_partial_attributes.shape[0],node_features.shape[1] + 1)).to(torch.float16)
    final_edge_attributes[:,0] = edge_partial_attributes.t()
    del edge_partial_attributes

    for batch_idx in pbar:


        # A batch example:
        #
        #  0      1        2      3        4      5        6      7
        # (A, B) (B, A)   (C, D) (D, C)   (E, F) (F, E)   (G, H) (H, G)

        for i in range(0, batch_size, 2):

            element_index = (batch_idx * batch_size) + i

            if element_index >= adjacency_list.shape[0]:
                break

            node1 = node_features[adjacency_list[element_index][0]]
            node2 = node_features[adjacency_list[element_index][1]]

            difference = (node1 - node2).to(torch.float16)

            final_edge_attributes[element_index, 1:] = difference
            final_edge_attributes[element_index + 1, 1:] = difference

        pass



    # GRAPH building

    graph = pyg_data.Data(
        detections=node_features,
        num_nodes=number_of_nodes,
        times=frame_times,
        edge_index=adjacency_list,
        edge_attr=final_edge_attributes,  # TODO: should edge attributes tensor have same size of the adjacency list?
    )

    return graph


class MotTrack:
    """Class used for a track in a dataset"""

    def __init__(self,
                 track_path: str,
                 detection_file_name: str,
                 images_directory: str,
                 det_resize: tuple,
                 linkage_window: int,
                 device: str):
        self.track_dir = track_path
        self.det_file = os.path.normpath(os.path.join(self.track_dir, "det", detection_file_name))
        self.img_dir = os.path.normpath(os.path.join(self.track_dir, images_directory))
        self.det_resize = det_resize
        self.linkage_window = linkage_window
        self.detections = self._read_detections()
        self.device = device
        self.n_nodes = None  # will be a list of int

        if self.det_resize is None:
            self.det_resize = self._get_avg_size()

    def _get_avg_size(self) -> tuple:
        ws, hs = [], []
        for frame_idx in self.detections.keys():
            for bbox in self.detections[frame_idx]:
                ws.append(bbox[2])
                hs.append(bbox[3])
        ws, hs = np.array(ws), np.array(hs)
        ws = ws[ws < np.percentile(ws, 95)]
        hs = hs[hs < np.percentile(hs, 95)]

        return tuple([
            int(ws.mean()),
            int(hs.mean())
        ])

    def _read_detections(self, delimiter=",", columns=(0, 2, 3, 4, 5)) -> dict:
        """Read detections into a dictionary"""

        file = np.loadtxt(self.det_file, delimiter=delimiter, usecols=columns)

        detections = {}
        for det in file:
            frame = int(det[0])
            if frame not in detections:
                detections[frame] = []
            detections[frame].append(det[1:].tolist())

        return detections

    def get_data(self, limit=-1, distance_power=2) -> tuple:
        """
        This function performs two steps:
        1) DETECTIONS EXTRACTION - extract all relevant features of all detections in all frames.
           (so far, only the center distance, other features will be added later)
        2) NODE LINKAGE - creates adjacency matrix and feature matrix, according to linkage type

        :param limit: limit the number of frames to process. -1 = no limit
        :param distance_power: power for the minkowski distance. NB: 1=Manhattan, 2=Euclidean. Must be >= 1
        :return: a tuple: (adjacency_matrix, flattened_node_features, frame_times, edge_attributes)
        """
        # # # # # # # # # # # # #
        # DETECTIONS EXTRACTION #
        # # # # # # # # # # # # #

        track_detections = []  # all image detections across all frames
        track_detections_centers = []  # all detections coordinates (xyxy) across all frames

        pbar = tqdm(sorted(os.listdir(self.img_dir)))

        i = 0  # frame counter

        for image in pbar:

            pbar.set_description(f"Reading frame {self.img_dir}/{image}")

            image = Image.open(os.path.normpath(os.path.join(self.img_dir, image)))

            frame_detections = []  # all image detections in the current frame
            frame_detections_centers = []  # all detection centers (0,1 scaled) in the current frame

            for j, bbox in enumerate(self.detections[i + 1]):
                bbox = box_convert(torch.tensor(bbox), "xywh", "xyxy").tolist()

                detection = image.crop(bbox)
                detection = detection.resize(self.det_resize)
                detection = transforms.PILToTensor()(detection)

                # Scale the bbox coordinates to the new image size
                bbox[0] *= self.det_resize[0] / image.size[0]
                bbox[1] *= self.det_resize[1] / image.size[1]
                bbox[2] *= self.det_resize[0] / image.size[0]
                bbox[3] *= self.det_resize[1] / image.size[1]

                # Get the [0,1] scaled bbox center
                bbox_center = [
                    (bbox[0] + bbox[2]) / 2 / self.det_resize[0],
                    (bbox[1] + bbox[3]) / 2 / self.det_resize[1]
                ]

                frame_detections.append(detection)
                frame_detections_centers.append(bbox_center)

            track_detections.append(frame_detections)
            track_detections_centers.append(frame_detections_centers)

            if i + 1 == limit:
                break

            i += 1

        # Flattened list of node features
        flattened_node_features = [item
                                   for sublist in track_detections
                                   for item in sublist]

        flattened_node_features = torch.stack(flattened_node_features).to(self.device)

        # List with the frame number of each detection
        frame_times = [i
                       for i in range(len(track_detections))
                       for _ in range(len(track_detections[i]))]

        frame_times = torch.tensor(frame_times).to(self.device)

        # # # # # # # # #
        # NODE LINKAGE  #
        # # # # # # # # #

        # Empty list to store edge attributes (will be converted to tensor)
        edge_attr = list()

        # Empty list to store edge indices
        adjacency_list = list()

        # List with the number of nodes per frame (aka number of detections per frame)
        self.n_nodes = [len(frame_detections) for frame_detections in track_detections]

        # Cumulative sum of the number of nodes per frame, used in the building of edge_index
        n_sum = [0] + np.cumsum(self.n_nodes).tolist()

        # LINKAGE Type No.0: ADJACENT
        if self.linkage_window == LINKAGE_TYPES["ADJACENT"]:

            #   Connect ADJACENT frames detections
            #   Edge features:
            #   - pairwise detections centers distance

            for i in tqdm(range(1, len(track_detections)), desc="Building edge attributes"):
                for j in range(len(track_detections[i])):
                    for k in range(len(track_detections[i - 1])):
                        # Add pairwise bbox centers distance
                        # https://stackoverflow.com/a/63196534

                        center_distance = torch.tensor([
                            minkowski_distance([track_detections_centers[i][j][0],
                                                track_detections_centers[i][j][1]],
                                               [track_detections_centers[i - 1][k][0],
                                                track_detections_centers[i - 1][k][1]],
                                               distance_power)
                        ])

                        edge_attr.append(center_distance)
                        edge_attr.append(center_distance)

                        # Build the adjacency list

                        adjacency_list.append([
                            j + n_sum[i],
                            k + n_sum[i - 1]
                        ])
                        adjacency_list.append([
                            k + n_sum[i - 1],
                            j + n_sum[i]
                        ])

                        pass  # debug

        # LINKAGE Type No.1
        elif self.linkage_window == LINKAGE_TYPES["ALL"]:

            #   Connect frames detections with ALL detections from different frames
            #   Edge features:
            #   - pairwise detections centers distance
            #   - pairwise detections time distance (frames)

            for i in tqdm(range(len(track_detections)), desc="Building edge attributes"): # for each frame in the track
                for j in range(len(track_detections[i])): # for each detection of that frame (i)
                    for k in range(i + 1, len(track_detections)): # for each frame in the track

                        # if k <= i:
                        #     continue  # skip same frame detections

                        for l in range(len(track_detections[k])): # for each detection of the frame (k)
                            center_distance = torch.tensor([
                                minkowski_distance(
                                    [track_detections_centers[i][j][0],
                                     track_detections_centers[i][j][1]],
                                    [track_detections_centers[k][l][0],
                                     track_detections_centers[k][l][1]],
                                    distance_power)
                            ])

                            edge_attr.append(center_distance)
                            edge_attr.append(center_distance)

                            adjacency_list.append([
                                # np.ushort(j + n_sum[i]),
                                # np.ushort(l + n_sum[k])
                                j + n_sum[i], l + n_sum[k]
                            ])
                            adjacency_list.append([
                                # np.ushort(l + n_sum[k]),
                                # np.ushort(j + n_sum[i])
                                l + n_sum[k], j + n_sum[i]
                            ])
        else: # Linkage window
            for i in tqdm(range(len(track_detections)), desc="Building edge attributes"): # for each frame in the track
                for j in range(len(track_detections[i])): # for each detection of that frame (i)
                    for k in range(len(track_detections)): # for each frame in the track

                        if k <= i:
                            continue  # skip same frame detections
                        if k > i + self.linkage_window:
                            break

                        for l in range(len(track_detections[k])): # for each detection of the frame (k)
                            center_distance = torch.tensor([
                                minkowski_distance(
                                    [track_detections_centers[i][j][0],
                                     track_detections_centers[i][j][1]],
                                    [track_detections_centers[k][l][0],
                                     track_detections_centers[k][l][1]],
                                    distance_power)
                            ])

                            edge_attr.append(center_distance)
                            edge_attr.append(center_distance)

                            adjacency_list.append([
                                # np.ushort(j + n_sum[i]),
                                # np.ushort(l + n_sum[k])
                                j + n_sum[i], l + n_sum[k]
                            ])
                            adjacency_list.append([
                                # np.ushort(l + n_sum[k]),
                                # np.ushort(j + n_sum[i])
                                l + n_sum[k], j + n_sum[i]
                            ])


        # Convert the edge attributes list to a tensor
        edge_attr = torch.stack(edge_attr)

        # Prepare the edge index tensor for pytorch geometric
        adjacency_list = torch.tensor(adjacency_list)  # .t().contiguous()

        print(f"[INFO] {len(track_detections)} total frames")
        print(f"[INFO] {len(flattened_node_features)} total nodes")
        print(f"[INFO] {len(adjacency_list)} total edges")

        # # # # # #
        # OUTPUT  #
        # # # # # #

        return adjacency_list, flattened_node_features, frame_times, edge_attr


class MotDataset(Dataset):

    def __init__(self,
                 dataset_path,
                 split,
                 detection_file_name="det.txt",
                 images_directory="img1",
                 det_resize=None,
                 linkage_window=0,
                 device="cpu"):
        self.dataset_dir = dataset_path
        self.split = split
        self.detection_file_name = detection_file_name
        self.images_directory = images_directory
        self.det_resize = det_resize
        self.device = device
        self.linkage_window = linkage_window

        assert split in os.listdir(self.dataset_dir), \
            f"Split must be one of {os.listdir(self.dataset_dir)}."

        self.tracklist = os.listdir(os.path.join(self.dataset_dir, split))

    def __getitem__(self, item):
        track_path = os.path.join(self.dataset_dir, self.split, self.tracklist[item])
        track_obj = MotTrack(track_path,
                             self.detection_file_name,
                             self.images_directory,
                             self.det_resize,
                             self.linkage_window,
                             self.device)

        return track_obj  # .get_data()
