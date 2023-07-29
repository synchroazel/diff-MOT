"""
Set of classes used to deal with datasets and tracks.
"""

import logging
import os

import numpy as np
import torch
import torch_geometric.data as pyg_data
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.transforms import KNNGraph
from torchvision import transforms
from torchvision.ops import box_convert
from tqdm import tqdm

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

LINKAGE_TYPES = {
    "ADJACENT": 0,
    "ALL": -1
}


class MOTGraph(pyg_data.Data):

    def __init__(self, y, times, gt_adjacency_list, detections, **kwargs):
        super().__init__(**kwargs)
        self.y = y
        self.times = times
        self.gt_edge_index = gt_adjacency_list
        self.detections = detections


def build_graph(adjacency_list: torch.Tensor,
                gt_adjacency_list: torch.Tensor,
                detections: torch.Tensor,
                frame_times: torch.Tensor,
                detections_coords: torch.Tensor,
                device: torch.device,
                dtype: torch.dtype = torch.float32,
                mps_fallback: bool = False,
                naive_pruning_args=None,
                knn_pruning_args=None) -> pyg_data.Data:
    """
    This function's purpose is to process the output of `track.get_data()` to build an appropriate graph.

    :param adjacency_list: tensor which represents the adjacency list. This will be used as 'edge_index'
    :param gt_adjacency_list: tensor which represents the ground truth adjacency list
    :param detections: tensor which represents the node image
    :param frame_times: time distances of the frames
    :param detections_coords: coordinates of the detections bboxes
    :param device: device to use, either mps, cuda or cpu
    :param dtype: data type to use
    :param mps_fallback: if True, will fallback to cpu on certain operations if not supported by MPS
    :param naive_pruning_args: args for naive pruning {"dist": int} - if None pruning is disabled
    :param knn_pruning_args: args for knn pruning {"k": int, "cosine": bool} - if None pruning is disabled
    :return: a Pytorch Geometric graph object
    """

    if knn_pruning_args is not None and naive_pruning_args is not None:
        raise ValueError("Cannot use multiple pruning methods at the same time.")

    detections = detections.to(dtype)
    number_of_nodes = len(detections)

    # For distance calculation
    detections_coords = box_convert(torch.clone(detections_coords).detach(), "xyxy", "cxcywh")

    # Assigned to data.pos, used for knn
    # position_matrix = torch.zeros((detections_coords.shape[0], 2))
    position_matrix = detections_coords[:, (0, 1)]

    # / 1000 is needed if we have 16bit floats, otherwise overflow will occur
    # detections_coords = detections_coords[:, (0, 1)] / 1000

    # detections_dist = torch.cdist(detections_coords.to(torch.float32), detections_coords.to(torch.float32)).to(dtype)

    # Prepare for pyg_data.Data
    adjacency_list = adjacency_list.t().contiguous()
    gt_adjacency_list = gt_adjacency_list.t().contiguous() if gt_adjacency_list is not None else None

    graph = MOTGraph(
        edge_index=adjacency_list,
        gt_adjacency_list=gt_adjacency_list,
        y=None,
        detections=detections,
        num_nodes=number_of_nodes,
        edge_attr=None,
        pos=position_matrix,
        times=frame_times
    )

    # kNN edge pruning
    if knn_pruning_args is not None:

        if mps_fallback:
            graph = graph.to('cpu')

        knn_morpher = KNNGraph(loop=False, force_undirected=True, **knn_pruning_args)
        graph = knn_morpher(graph)

        if mps_fallback:
            graph = graph.to('mps')

    edge_attributes = torch.zeros(graph.edge_index.shape[1], 2)

    # TODO: SUSHI attributes
    # # once the graph is pruned, compute edge attributes
    # edge_attributes = torch.zeros(graph.edge_index.shape[1], 7).to(device)
    # # obtain info for each edge
    # x = list()
    # y = list()
    # h = list()
    # w = list()
    # t = list()
    # GIoU = list()
    # Gboxes = box_convert(detections_coords, "cxcywh", "xyxy")
    # for egde in graph.edge_index.t():
    #     x.append(
    #         ((2*(detections_coords[egde[0],0] - detections_coords[egde[1],0])) /
    #         (detections_coords[egde[0],2] + detections_coords[egde[1],2])).item()
    #     )
    #     y.append(
    #         ((2 * (detections_coords[egde[0], 1] - detections_coords[egde[1], 1])) /
    #         (detections_coords[egde[0],3] + detections_coords[egde[1], 3])).item()
    #     )
    #     h.append(torch.log(detections_coords[egde[0],2] / detections_coords[egde[1],2]).item())
    #     w.append(torch.log(detections_coords[egde[0],3] / detections_coords[egde[1],3]).item())
    #     t.append((frame_times[egde[1]] - frame_times[egde[0]]).item() / frame_times[-1])
    #     GIoU.append(
    #         generalized_box_iou(
    #             boxes1=Gboxes[egde[0],:].unsqueeze(0),
    #             boxes2=Gboxes[egde[1],:].unsqueeze(0)
    #         ).item()
    #     )
    #
    # # position information
    # edge_attributes[:,0] = torch.tensor(x)
    # edge_attributes[:,1] = torch.tensor(y)
    # edge_attributes[:, 2] = torch.tensor(h)
    # edge_attributes[:, 3] = torch.tensor(w)
    # # Time information
    # edge_attributes[:, 4] = torch.tensor(t)
    #
    # # Appearance information
    #
    #
    # # Motion consistency information
    # edge_attributes[:,6] = torch.tensor(GIoU)

    """ Our edge attributes """

    edge_attributes = torch.zeros(graph.edge_index.shape[1], 2).to(device)

    # Those 2 are calculated here because knn_morpher does not update them
    detections_dist = torch.cdist(graph.pos,
                                  graph.pos, p=2).to('cpu')

    # Create a 1D Tensor with the distances of the detections ordered as the edges in the adj list
    a = graph.edge_index.t()[:, 0].to('cpu').to(torch.int64)
    b = graph.edge_index.t()[:, 1].to('cpu').to(torch.int64)
    edge_attributes[:, 0] = detections_dist[a, b]

    # Compute the spatial distance between the detections

    detections_dist = torch.cdist(frame_times.to(dtype),
                                  frame_times.to(dtype), p=2).to('cpu')

    # Create a 1D Tensor with the distances of the detections ordered as the edges in the adj list
    edge_attributes[:, 1] = detections_dist[a, b]

    # Delete the tensors to free up memory
    del detections_dist
    del a
    del b

    # edge_attributes = 1 / (edge_attributes.to(device) + 0.00001)
    graph.edge_attr = edge_attributes

    # Build `y` tensor to compare predictions with gt
    if gt_adjacency_list is not None:
        gt_adjacency_set = set([tuple(x) for x in gt_adjacency_list.t().tolist()])
        y = torch.tensor([1 if tuple(x) in gt_adjacency_set else 0 for x in graph.edge_index.t().tolist()]).to(dtype)
        graph.y = y

    # # Naive edge pruning - distance of 20 pixel per time
    # if naive_pruning_args is not None:
    #     pruned_mask = [False if x[0] < (1 / naive_pruning_args['dist']) * x[1] else True for x in edge_attributes]
    #     graph.edge_index = adjacency_list[:, pruned_mask]

    return graph


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
                 name: str = "track",
                 black_and_white_features=False):

        # Set logging level
        logging.getLogger().setLevel(logging_lv)

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
        self.black_and_white_features = black_and_white_features

        logging.info(f"{self.n_frames} frames")

        # CHeck if the chosen linkage window is possible
        if self.linkage_window > self.n_frames:
            logging.warning(f"`linkage window` was set to {self.linkage_window} but track has {self.n_frames} frames."
                            f"Setting `linkage window` to {self.n_frames}")
            self.linkage_window = self.n_frames

    # # OLD __str__ method:
    #
    # def __str__(self):
    #
    #     name = self.name
    #
    #     if self.linkage_window == -1 and self.subtrack_len == -1:
    #         name += "/" + self.name.split("/")[-1]
    #
    #     if self.linkage_window > 0:
    #         name += "/window_" + str(self.linkage_window)
    #
    #     if self.subtrack_len > 0:
    #         name += "_len_" + str(self.subtrack_len)
    #
    #     return name

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
        channels = 3
        if self.black_and_white_features:
            channels = 1

        number_of_detections = sum([len(x) for x in self.detections])

        track_detections_coords = torch.zeros((number_of_detections, 4), dtype=self.dtype).to(self.device)
        frame_times = torch.zeros((number_of_detections, 1), dtype=torch.int16).to(self.device)
        flattened_node_features = torch.zeros((number_of_detections, channels, self.det_resize[1], self.det_resize[0]),
                                              dtype=self.dtype).to(self.device)

        # Process all frames images
        for image in pbar:
            nodes = 0

            if self.logging_lv <= logging.INFO:
                pbar.set_description(f"[TQDM] Reading frame {image}")

            image = Image.open(os.path.normpath(image))  # all image detections in the current frame
            if self.black_and_white_features:
                image = image.convert('L')

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
                        adjacency_list.append([l, j]) # ----------------------------------------------------------------

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

        logging.info(f"{len(flattened_node_features)} total nodes")
        logging.info(f"{len(adjacency_list)} total edges")

        """
        Node linkage - ground truth
        """

        gt_adjacency_list = None

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

            # Fill gt_adjacency_list using the detections in all_paths
            for path in all_paths:
                for i in range(len(path) - 1):
                    gt_adjacency_list.append([path[i], path[i + 1]])
                    gt_adjacency_list.append([path[i + 1], path[i]]) # -------------------------------------------------

            # Prepare the edge index tensor for pytorch geometric
            gt_adjacency_list = torch.tensor(gt_adjacency_list).to(torch.int16).to(self.device)

            logging.info(f"{len(gt_adjacency_list)} total gt edges ({len(all_paths)} trajectories)")

        else:
            logging.info(f"No ground truth adjacency list available")

        """
        Output section
        """

        return {
            "adjacency_list": adjacency_list,
            "gt_adjacency_list": gt_adjacency_list,
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
                 slide: int = 1,
                 dl_mode: bool = False,
                 black_and_white_features=False,
                 naive_pruning_args=None,
                 knn_pruning_args=None,
                 mps_fallback: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype=torch.float32):
        self.dataset_dir = dataset_path
        self.split = split
        self.detections_file_folder = detections_file_folder
        self.detections_file_name = detections_file_name
        self.images_directory = images_directory
        self.det_resize = det_resize
        self.linkage_window = linkage_window
        self.slide = slide
        self.subtrack_len = subtrack_len
        self.mps_fallback = mps_fallback
        self.black_and_white_features = black_and_white_features
        self.device = device
        self.dl_mode = dl_mode
        self.dtype = dtype
        self.name = dataset_path.split('/')[-1] if name is None else name
        self.frames_per_track = None
        self.n_subtracks = None
        self.cur_track = None
        self.str_frame = None
        self.end_frame = None
        self.start_frames = None
        self.tracks = None

        # Initialization for pruning arguments
        if naive_pruning_args is None:
            self.naive_pruning_args = None  # {"dist": 20}
        if knn_pruning_args is None:
            self.knn_pruning_args = {"k": 10, "cosine": False}
        else:
            self.knn_pruning_args = knn_pruning_args

        # Get tracklist from the split specified
        assert split in os.listdir(self.dataset_dir), \
            f"Split must be one of {os.listdir(self.dataset_dir)}."
        self.tracklist = os.listdir(os.path.join(self.dataset_dir, split))

        # Set the logging level according to the use we're doing of the dataset
        if self.dl_mode:
            logging.getLogger().setLevel(logging.WARNING)

        # Precompute if slide or subtrack_len is provided
        if self.slide > 1 or self.subtrack_len > 1:
            self.precompute_subtracks()

    def precompute_subtracks(self):
        frames_per_track = []
        for track in self.tracklist:
            track_path = os.path.join(self.dataset_dir, self.split, track)
            img_dir = os.path.normpath(os.path.join(track_path, self.images_directory))
            images_list = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
            frames_per_track.append(len(images_list))

        self.start_frames = []
        self.tracks = []
        for track, n_frames in enumerate(frames_per_track):
            for start_frame in range(0, n_frames - self.subtrack_len + 1, self.slide):
                self.start_frames.append(start_frame)
                self.tracks.append(track)

        if len(self.start_frames) == 0:
            raise ValueError(f"No subtracks of len {self.subtrack_len} can be created with slide {self.slide}.")

        self.frames_per_track = frames_per_track
        self.n_subtracks = len(self.start_frames)
        self.cur_track = self.tracks[0]
        self.str_frame = self.start_frames[0]
        self.end_frame = self.str_frame + self.subtrack_len

    @staticmethod
    def _read_detections(det_file) -> dict:
        """Read detections into a dictionary"""

        file = np.loadtxt(det_file, delimiter=",")

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
        if idx >= self.n_subtracks:
            raise IndexError(
                f"At most {self.n_subtracks} subtracks of len {self.subtrack_len} can be created with slide {self.slide}.")

        all_detections = []
        for track in self.tracklist:
            track_path = os.path.join(self.dataset_dir, self.split, track)
            detections_file = os.path.normpath(
                os.path.join(track_path, self.detections_file_folder, self.detections_file_name))
            detections = self._read_detections(detections_file)
            detections = [detections[frame] for frame in sorted(detections.keys())]
            all_detections += [detections]

        all_images = []
        for track in self.tracklist:
            track_path = os.path.join(self.dataset_dir, self.split, track)
            img_dir = os.path.normpath(os.path.join(track_path, self.images_directory))
            images_list = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
            all_images += [images_list]

        # Get the starting frame and track for this batch
        starting_frame = self.start_frames[idx]
        cur_track = self.tracks[idx]
        ending_frame = starting_frame + self.subtrack_len

        logging.debug(f"From {starting_frame} to {ending_frame}")
        logging.debug(f"Starting in track: {cur_track}")
        logging.debug(f"From {starting_frame} to {ending_frame} of track {cur_track}")

        frames_window_msg = f"frames {starting_frame}-{ending_frame}/{len(all_images[cur_track])}"

        logging.info(
            f"Subtrack #{idx} | track {self.tracklist[cur_track]} {frames_window_msg}\r")

        self.cur_track = cur_track
        self.str_frame = starting_frame
        self.end_frame = ending_frame

        track = MotTrack(detections=all_detections[cur_track][starting_frame:ending_frame],
                         images_list=all_images[cur_track][starting_frame:ending_frame],
                         det_resize=self.det_resize,
                         linkage_window=self.linkage_window,
                         subtrack_len=self.subtrack_len,
                         black_and_white_features=self.black_and_white_features,
                         device=self.device,
                         dtype=self.dtype,
                         logging_lv=logging.WARNING if self.dl_mode else logging.INFO,
                         name=self.name + "/track_" + str(self.tracklist[cur_track]) + "/subtrack_" + str(idx))

        if self.dl_mode:
            return build_graph(
                mps_fallback=self.mps_fallback,
                device=self.device,
                dtype=self.dtype,
                naive_pruning_args=self.naive_pruning_args,
                knn_pruning_args=self.knn_pruning_args,
                **track.get_data())
        else:
            return track
