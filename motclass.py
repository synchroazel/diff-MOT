"""Set of classes used to deal with datasets and tracks"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import box_convert
from tqdm import tqdm

LINKAGE_TYPES = {
    "ADJACENT": 0,
    "ALL": 1
}

class MotTrack:
    """Class used for a track in a dataset"""

    def __init__(self,
                 track_path: str,
                 detection_file_name: str,
                 images_directory: str,
                 det_resize: tuple,
                 linkage_type: str,
                 device: str):
        self.track_dir = track_path
        self.det_file = os.path.normpath(os.path.join(self.track_dir, "det", detection_file_name))
        self.img_dir = os.path.normpath(os.path.join(self.track_dir, images_directory))
        self.det_resize = det_resize
        self.linkage_type = LINKAGE_TYPES[linkage_type]
        self.detections = self._read_detections()
        self.device = device
        self.n_nodes = None # will be a list of int

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

    def get_data(self, limit=-1) -> tuple:
        """
        This function performs two steps:
        1) DETECTIONS EXTRACTION - extract all relevant features of all detections in all frames
        2) NODE LINKAGE - creates adjacency matrix and feature matrix, according to linkage type
        :param limit: limit the number of frames to process. -1 = no limit
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

            pbar.set_description(f"Processing frame #{i + 1} ({image})")

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
        adjacency_matrix = list()

        # List with the number of nodes per frame (aka number of detections per frame)
        self.n_nodes = [len(frame_detections) for frame_detections in track_detections]

        # Cumulative sum of the number of nodes per frame, used in the building of edge_index
        n_sum = [0] + np.cumsum(self.n_nodes).tolist()

        # LINKAGE Type No.0: ADJACENT
        if self.linkage_type == 0:

            #   Connect ADJACENT frames detections
            #   Edge features:
            #   - pairwise detections centers distance
            #   - pairwise detections features difference TODO: implement

            for i in range(1, len(track_detections)):
                for j in range(len(track_detections[i])):
                    for k in range(len(track_detections[i - 1])):
                        # Add pairwise bbox centers distance
                        edge_attr.append(torch.tensor([
                            track_detections_centers[i][j][0] - track_detections_centers[i - 1][k][0],
                            track_detections_centers[i][j][1] - track_detections_centers[i - 1][k][1]
                        ]))

                        # Build the adjacency matrix

                        adjacency_matrix.append([
                            j + n_sum[i],
                            k + n_sum[i - 1]
                        ])
                        adjacency_matrix.append([
                            k + n_sum[i - 1],
                            j + n_sum[i]
                        ])

                        pass # debug

        # LINKAGE Type No.1
        elif self.linkage_type == 1:

            #   Connect frames detections with ALL detections from different frames
            #   Edge features:
            #   - pairwise detections centers distance
            #   - pairwise detections features difference TODO: implement
            #   - pairwise detections time distance (frames)

            for i in range(len(track_detections)):
                for j in range(len(track_detections[i])):
                    for k in range(len(track_detections)):
                        if k == i:
                            continue  # skip same frame detections

                        for l in range(len(track_detections[k])):
                            edge_attr.append(torch.tensor([
                                track_detections_centers[i][j][0] - track_detections_centers[i - 1][k][0],
                                track_detections_centers[i][j][1] - track_detections_centers[i - 1][k][1]
                            ]))

                            adjacency_matrix.append([
                                j + n_sum[i],
                                l + n_sum[k]
                            ])
                            adjacency_matrix.append([
                                l + n_sum[k],
                                j + n_sum[i]
                            ])

        # Convert the edge attributes list to a tensor
        edge_attr = torch.stack(edge_attr)

        # Prepare the edge index tensor for pytorch geometric
        adjacency_matrix = torch.tensor(adjacency_matrix)  # .t().contiguous()

        print(f"[INFO] {len(adjacency_matrix)} total edges")

        # # # # # #
        # OUTPUT  #
        # # # # # #

        # graph = pyg_data.Data(
        #     detections=flattened_node_features,
        #     num_nodes=flattened_node_features.shape[0],
        #     times=frame_times,
        #     edge_index=adjacency_matrix,
        #     edge_attr=edge_attr,
        # )
        #
        # return graph

        return adjacency_matrix, flattened_node_features, frame_times, edge_attr


class MotDataset(Dataset):

    def __init__(self,
                 dataset_path,
                 split,
                 detection_file_name="det.txt",
                 images_directory="img1",
                 det_resize=None,
                 linkage_type="ADJACENT",
                 device="cpu"):
        self.dataset_dir = dataset_path
        self.split = split
        self.detection_file_name = detection_file_name
        self.images_directory = images_directory
        self.det_resize = det_resize
        self.device = device
        self.linkage_type = linkage_type

        assert split in os.listdir(self.dataset_dir), \
            f"Split must be one of {os.listdir(self.dataset_dir)}."

        self.tracklist = os.listdir(os.path.join(self.dataset_dir, split))

        if self.det_resize is None:
            self.det_resize = self._get_avg_size()

    def _get_avg_size(self):

        width = list()
        height = list()

        for track in self:
            for f in os.listdir(track.img_dir):
                im = Image.open(os.path.join(track.img_dir, f))
                width.append(im.size[0])
                height.append(im.size[1])

        return tuple([
            int(np.mean(width)),
            int(np.mean(height))
        ])

    def __getitem__(self, item):
        track_path = os.path.join(self.dataset_dir, self.split, self.tracklist[item])
        track_obj = MotTrack(track_path,
                             self.detection_file_name,
                             self.images_directory,
                             self.det_resize,
                             self.linkage_type,
                             self.device)

        return track_obj  # .get_data()
