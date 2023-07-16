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

import logging

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

LINKAGE_TYPES = {
    "ADJACENT": 0,
    "ALL": -1
}


# adjacency_list, flattened_node_features, frame_times, centers_coords

def build_graph(adjacency_list: torch.Tensor,
                gt_adjacency_list: torch.Tensor,
                y: torch.Tensor,
                detections: torch.Tensor,
                frame_times: torch.Tensor,
                detections_coords: torch.Tensor,
                device: torch.device,
                dtype: torch.dtype = torch.float32) -> pyg_data.Data:
    """
    This function's purpose is to process the output of `track.get_data()` to build an appropriate graph.

    :param adjacency_list: tensor which represents the adjacency list. This will be used as 'edge_index'
    :param gt_adjacency_list: tensor which represents the ground truth adjacency list
    :param y
    :param detections: tensor which represents the node image
    :param frame_times: time distances of the frames
    :param detections_coords: coordinates of the detections bboxes
    :param device: device to use, either mps, cuda or cpu
    :param dtype: data type to use
    :return: a Pytorch Geometric graph object
    """

    batch_size = 8  # used throughout edges and nodes processing

    detections = detections.to(dtype)
    number_of_nodes = len(detections)

    # for distance calculation
    detections_coords = box_convert(torch.clone(detections_coords).detach(), "xyxy", "cxcywh")

    # centers = torch.zeros((len(detections_coords),2), dtype=dtype, device=device)
    # centers[:,(0,1)] = detections_coords[:,(0,1)] / 1000
    # del detections_coords
    detections_coords = detections_coords[:, (0, 1)] / 1000

    """
    EDGES processing - edge features computation
    """

    """ Detection features difference [ON HOLD]

    # Precompute the # of batches
    n_batches = int(np.ceil(adjacency_list.shape[0] / batch_size))

    # Precompute edge features, `vstack` causes memory leak
    final_edge_attributes = torch.zeros((edge_partial_attributes.shape[0], detections.shape[1] + 1),
                                        dtype=torch.float16)
    final_edge_attributes[:, 0] = edge_partial_attributes.t()
    del edge_partial_attributes

    pbar = tqdm(range(0, int(n_batches)), desc="Processing edges")

    for batch_idx in pbar:

        for i in range(0, batch_size, 2):

            element_index = (batch_idx * batch_size) + i

            if element_index >= adjacency_list.shape[0]:
                break

            node1 = detections[adjacency_list[element_index][0]]
            node2 = detections[adjacency_list[element_index][1]]

            difference = (node1 - node2).to(torch.float16)

            final_edge_attributes[element_index, 1:] = difference
            final_edge_attributes[element_index + 1, 1:] = difference

        pass
        
    """

    """ Detections distance """

    # detections_dist = torch.cdist(detections_coords.to(torch.float32), detections_coords.to(torch.float32)).to(dtype)

    # torch.cdist is not currently implemented on MPS - fallback to cpu in case of MPS device
    # if device == torch.device("mps"):
    #     detections_coords = detections_coords.float().cpu()

    detections_dist = torch.cdist(detections_coords,
                                  detections_coords, p=2).to('cpu')

    # Create a 1D Tensor with the distances of the detections ordered as the edges in the adj list
    a = adjacency_list[:, 0].to('cpu').to(torch.int64)
    b = adjacency_list[:, 1].to('cpu').to(torch.int64)
    partial_edge_features = detections_dist[a, b]
    # input("after indexing")
    del detections_dist
    del a
    del b
    # input("detection dist deleted")

    partial_edge_features = partial_edge_features.to(device) * 1000
    # print(partial_edge_features)
    ###
    # GRAPH building
    ###
    # input("edges moved to gpu")

    # Prepare for pyg_data.Data
    adjacency_list = adjacency_list.t().contiguous()
    gt_adjacency_list = gt_adjacency_list.t().contiguous() if gt_adjacency_list is not None else None

    return pyg_data.Data(
        edge_index=adjacency_list,
        gt_edge_index=gt_adjacency_list,
        y=y,
        detections=detections,
        num_nodes=number_of_nodes,
        times=frame_times,
        edge_features=partial_edge_features
    )


class MotTrack:
    """ Class representing a track in a MOT dataset """

    def __init__(self,
                 detections: list,
                 images_list: list,
                 det_resize: tuple,
                 linkage_window: int,
                 subtrack_len: int,
                 device: torch.device,
                 dtype: torch.dtype = torch.float32,
                 logging_lv: int = logging.INFO,
                 name: str = "track"):

        # Set logging level
        logging.basicConfig(level=logging_lv)

        self.detections = detections
        self.images_list = images_list
        self.det_resize = det_resize
        self.linkage_window = linkage_window
        self.subtrack_len = subtrack_len
        self.device = device
        self.name = name
        self.n_frames = len(images_list)
        self.n_nodes = []
        self.dtype = dtype
        self.logging_lv = logging_lv

        if self.linkage_window > self.n_frames:
            logging.warning(f"`linkage window` was set to {self.linkage_window} but track has {self.n_frames} frames."
                            f"Setting `linkage window` to {self.n_frames}")
            self.linkage_window = self.n_frames

        logging.info(f"Track has {self.n_frames} frames")

    def __str__(self):

        name = self.name

        if self.linkage_window == -1 and self.subtrack_len == -1:
            name += "/" + self.name.split("/")[-1]

        if self.linkage_window > 0:
            name += "/window_" + str(self.linkage_window)

        if self.subtrack_len > 0:
            name += "_len_" + str(self.subtrack_len)

        return name

    def get_data(self):
        """
        This function performs two steps:

        1) DETECTIONS EXTRACTION
           Detections are extracted, cropped, resized and turned to Tensor.

        2) NODE LINKAGE
           Creates adjacency matrix and feature matrix, according to a linkage type.

        :return: a dict: with keys (adjacency_matrix, node_features, frame_times, detections_coords)
        """

        """
        Detections processing
        """

        pbar = tqdm(self.images_list) \
            if self.logging_lv <= logging.INFO else self.images_list

        i = 0  # frame counter
        j = 0

        number_of_detections = sum([len(x) for x in self.detections])

        track_detections_coords = torch.zeros((number_of_detections, 4), dtype=self.dtype).to(self.device)
        frame_times = torch.zeros((number_of_detections, 1), dtype=torch.int16).to(self.device)
        flattened_node_features = torch.zeros((number_of_detections, 3, self.det_resize[1], self.det_resize[0]),
                                              dtype=self.dtype).to(self.device)

        # Process all frames images
        for image in pbar:
            nodes = 0

            if self.logging_lv <= logging.INFO:
                pbar.set_description(f"[TQDM] Reading frame {image}")

            image = Image.open(os.path.normpath(image))  # all image detections in the current frame

            for detection in self.detections[i]:
                nodes += 1
                bbox = box_convert(torch.tensor(detection['bbox']), "xywh", "xyxy").tolist()

                detection = image.crop(bbox)
                detection = detection.resize(self.det_resize)
                detection = transforms.PILToTensor()(detection)

                flattened_node_features[j, :, :, :] = detection

                track_detections_coords[j, :] = torch.tensor(bbox)
                frame_times[j, 0] = i
                j += 1

            # List with the number of nodes per frame (aka number of detections per frame)
            self.n_nodes.append(nodes)
            i += 1

        """
        Node linkage
        """

        # `linkage_window` determines the type of linkage between detections.

        # LINKAGE TYPE: ALL (`linkage_window` = -1)
        #   Connect frames detections with ALL detections from different frames

        # LINKAGE TYPE: ADJACENT (`linkage_window` = 0)
        #   Connect ADJACENT frames detections

        # LINKAGE TYPE: WINDOW (`linkage_window` > 0)
        #   Connect frames detections with detections up to `linkage_window` frames in the future

        adjacency_list = list()  # Empty list to store edge indices
        n_sum = [0] + np.cumsum(self.n_nodes).tolist()

        pbar = tqdm(range(self.n_frames), desc="[TQDM] Linking all nodes") \
            if self.logging_lv <= logging.INFO else range(self.n_frames)

        for i in pbar:
            current_n_sum = n_sum[i]
            current_frame_nodes = self.n_nodes[i]
            current_frame_nodes_indices = range(current_n_sum, current_n_sum + current_frame_nodes)

            # Get the range of frames to link based on the linkage window
            if self.linkage_window == -1:  # Link with all future frames
                future_frames_range = range(i + 1, self.n_frames)
            elif self.linkage_window == 0:  # Link with the next frame only
                future_frames_range = range(i + 1, min(i + 2, self.n_frames))
            else:  # Link with a window of future frames
                future_frames_range = range(i + 1, min(i + self.linkage_window + 1, self.n_frames))

            for k in future_frames_range:
                future_n_sum = n_sum[k]
                future_frame_nodes = self.n_nodes[k]
                future_frame_nodes_indices = range(future_n_sum, future_n_sum + future_frame_nodes)

                # For each combination of current node and future node, create an edge
                for j in current_frame_nodes_indices:
                    for l in future_frame_nodes_indices:
                        adjacency_list.append([j, l])
                        adjacency_list.append([l, j])

        adjacency_list = torch.tensor(adjacency_list).to(torch.int16).to(self.device)

        # adjacency_list = list()  # Empty list to store edge indices
        #
        # # Cumulative sum of the number of nodes per frame, used in the building of edge_index
        # n_sum = [0] + np.cumsum(self.n_nodes).tolist()
        #
        # for i in tqdm(range(self.n_frames), desc="Linking all nodes"):  # for each frame in the track
        #     for j in range(self.n_nodes[i]):  # for each detection of that frame (i)
        #         for k in range(self.n_frames):  # for each frame in the track
        #
        #             if self.linkage_window == -1:
        #                 if k <= i:
        #                     continue
        #
        #             if self.linkage_window == 0:
        #                 if k != i + 1:
        #                     continue
        #
        #             if self.linkage_window > 0:
        #                 if k <= i:
        #                     continue
        #                 if k > i + self.linkage_window:
        #                     break
        #
        #             for l in range(self.n_nodes[k]):  # for each detection of the frame (k)
        #
        #                 adjacency_list.append(
        #                     [j + n_sum[i], l + n_sum[k]]
        #                 )
        #                 adjacency_list.append(
        #                     [l + n_sum[k], j + n_sum[i]]
        #                 )
        #
        # # Prepare the edge index tensor for pytorch geometric
        # adjacency_list = torch.tensor(adjacency_list).to(torch.int16).to(self.device)

        logging.info(f"{self.n_frames} frames")
        logging.info(f"{len(flattened_node_features)} total nodes")
        logging.info(f"{len(adjacency_list)} total edges")

        """
        Node linkage - ground truth
        """

        gt_adjacency_list = None
        y = None

        # If detections ids are available (!= -1)
        if self.detections[0][0]['id'] != -1:

            gt_adjacency_list = list()

            # Get list of all gt detections ids
            gt_detections_ids = list(set([detection['id'] for frame in self.detections for detection in frame]))

            all_paths = list()

            # Iterate over all the detections ids (aka over all gr trajectories)
            for det_id in gt_detections_ids:

                cur_path = list()

                # Iterate over all frames and all detections in each
                for i in range(self.n_frames):
                    for j in range(self.n_nodes[i]):

                        # Simply add to a list the # of the detection with that id
                        if self.detections[i][j]['id'] == det_id:
                            cur_path.append(j + n_sum[i])

                # all_paths wll be a list of lists, each list containing the detections ids of a gt trajectory
                all_paths.append(cur_path)

            logging.info(f"{len(all_paths)} gt trajectories found")

            # Fill gt_adjacency_list using the detections in all_paths
            for path in all_paths:
                for i in range(len(path) - 1):
                    gt_adjacency_list.append([path[i], path[i + 1]])
                    gt_adjacency_list.append([path[i + 1], path[i]])

            # Prepare the edge index tensor for pytorch geometric
            gt_adjacency_list = torch.tensor(gt_adjacency_list).to(torch.int16).to(self.device)

            logging.info(f"{len(gt_adjacency_list)} total gt edges ({len(all_paths)} trajectories)")

            # Build `y` tensor to compare predictions with gt
            gt_adjacency_set = set([tuple(x) for x in gt_adjacency_list.tolist()])

            y = torch.tensor([1 if tuple(x) in gt_adjacency_set else 0 for x in adjacency_list]).to(self.dtype)

        else:
            logging.info(f"No ground truth adjacency list available")

        """
        Output section
        """

        return {
            "adjacency_list": adjacency_list,
            "gt_adjacency_list": gt_adjacency_list,
            "y": y,
            "detections": flattened_node_features,
            "frame_times": frame_times,
            "detections_coords": track_detections_coords
        }


class MotDataset(Dataset):

    def __init__(self,
                 dataset_path: str,
                 split: str,
                 detections_file_folder: str = "gt",
                 detections_file_name: str = "gt.txt",
                 images_directory: str = "img1",
                 name: str = None,
                 det_resize: tuple = (70, 170),
                 linkage_window: int = -1,
                 subtrack_len: int = -1,
                 subtrack_number: int = -1,
                 dl_mode: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype=torch.float32):
        self.dataset_dir = dataset_path
        self.split = split
        self.detections_file_folder = detections_file_folder
        self.detections_file_name = detections_file_name
        self.images_directory = images_directory
        self.det_resize = det_resize
        self.linkage_window = linkage_window
        self.subtrack_len = subtrack_len
        self.subtrack_number = subtrack_number
        self.device = device
        self.dl_mode = dl_mode
        self.dtype = dtype
        self.name = dataset_path.split('/')[-1] if name is None else name

        assert split in os.listdir(self.dataset_dir), \
            f"Split must be one of {os.listdir(self.dataset_dir)}."

        self.tracklist = os.listdir(os.path.join(self.dataset_dir, split))

    @staticmethod
    def _read_detections(det_file) -> dict:
        """Read detections into a dictionary"""

        file = np.loadtxt(det_file, delimiter=",")  # , usecols=tuple(range(5)))

        detections = {}
        for det in file:
            frame = int(det[0])
            id = int(det[1])
            bbox = det[2:6].tolist()
            if frame not in detections:
                detections[frame] = []

            detections[frame].append({"id": id, "bbox": bbox})

        return detections

    def __getitem__(self, idx):

        frames_per_track, all_detections, all_images = list(), list(), list()

        for track in self.tracklist:
            track_path = os.path.join(self.dataset_dir, self.split, track)
            detections_file = os.path.normpath(
                os.path.join(track_path, self.detections_file_folder, self.detections_file_name))
            detections = self._read_detections(detections_file)
            detections = [detections[frame] for frame in sorted(detections.keys())]
            all_detections += [detections]

        all_images = list()

        for track in self.tracklist:
            track_path = os.path.join(self.dataset_dir, self.split, track)
            img_dir = os.path.normpath(os.path.join(track_path, self.images_directory))
            images_list = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
            frames_per_track.append(len(images_list))
            all_images += [images_list]

        # IF NO subtrack_len is provided, RETURN THE WHOLE TRACK

        if self.subtrack_len == -1:
            return MotTrack(detections=all_detections[idx],
                            images_list=all_images[idx],
                            det_resize=self.det_resize,
                            linkage_window=self.linkage_window,
                            subtrack_len=self.subtrack_len,
                            device=self.device,
                            dtype=self.dtype,
                            name=self.name + "/track_" + str(self.tracklist[idx]))

        # IF subtrack_len is provided, RETURN THE #idx BATCH

        if self.subtrack_len > 0:

            # Precompute the start frames and tracks if they haven't been computed already
            if not hasattr(self, 'start_frames') or not hasattr(self, 'tracks'):
                self.start_frames = []
                self.tracks = []
                for track, n_frames in enumerate(frames_per_track):
                    for start_frame in range(n_frames - self.subtrack_len + 1):
                        self.start_frames.append(start_frame)
                        self.tracks.append(track)

            if idx >= len(self.start_frames):
                raise IndexError(
                    f"{len(self.start_frames)} subtracks can be created with subtrack_len={self.subtrack_len}, but batch #{idx} was requested. ")

            # Get the starting frame and track for this batch
            starting_frame = self.start_frames[idx]
            cur_track = self.tracks[idx]
            ending_frame = starting_frame + self.subtrack_len

            logging.debug(f"From {starting_frame} to {ending_frame}")

            logging.debug(f"Starting in track: {cur_track}")
            logging.debug(f"From {starting_frame} to {ending_frame} of track {cur_track}")

            logging.info(
                f"Batch #{idx} | track #{cur_track + 1} (frames {starting_frame}/{frames_per_track[cur_track]} - {ending_frame}/{frames_per_track[cur_track]})")

            track = MotTrack(detections=all_detections[cur_track][starting_frame:ending_frame],
                             images_list=all_images[cur_track][starting_frame:ending_frame],
                             det_resize=self.det_resize,
                             linkage_window=self.linkage_window,
                             subtrack_len=self.subtrack_len,
                             device=self.device,
                             dtype=self.dtype,
                             logging_lv=logging.WARNING if self.dl_mode else logging.INFO,
                             name=self.name + "/track_" + str(self.tracklist[cur_track]) + "/subtrack_" + str(idx))

            if self.dl_mode:
                return build_graph(**track.get_data(), device=self.device, dtype=self.dtype)
            else:
                return track
