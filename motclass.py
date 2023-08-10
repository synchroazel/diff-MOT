"""
Set of classes used to deal with datasets and tracks.
"""
import torch
import torch_geometric.data as pyg_data
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.transforms import KNNGraph
from torchvision import transforms
from torchvision.ops import box_convert
from tqdm import tqdm

from model import ImgEncoder
from utilities import *

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


# TODO: fuse with previous train

class MOTGraph(pyg_data.Data):

    def __init__(self, y, times, gt_adjacency_dict, detections, **kwargs):
        super().__init__(**kwargs)
        self.y = y
        self.times = times
        self.gt_edge_indexes = gt_adjacency_dict
        self.detections = detections


def build_graph(linkage_window: int,
                n_nodes:list,
                gt_dict: dict,
                detections: torch.Tensor,
                frame_times: torch.Tensor,
                detections_coords: torch.Tensor,
                device: torch.device,
                dtype: torch.dtype = torch.float32,
                mps_fallback: bool = False,
                knn_pruning_args=None) -> pyg_data.Data:
    """
    This function's purpose is to process the output of `track.get_data()` to build an appropriate graph.

    :param adjacency_list: tensor which represents the adjacency list. This will be used as 'edge_index'
    :param
    :param detections: tensor which represents the node image
    :param frame_times: time distances of the frames
    :param detections_coords: coordinates of the detections bboxes
    :param device: device to use, either mps, cuda or cpu
    :param dtype: data type to use
    :param mps_fallback: if True, will fallback to cpu on certain operations if not supported by MPS
    :param knn_pruning_args: args for knn pruning {"k": int, "cosine": bool} - if None pruning is disabled
    :return: a Pytorch Geometric graph object
    """

    positions = box_convert(torch.clone(detections_coords).detach(), "xyxy", "cxcywh")
    position_matrix = positions[:, (0, 1)]

    # position_matrix = detections

    distance_matrix = torch.cdist(detections, detections)

    detections = detections.to(dtype)
    number_of_nodes = len(detections)
    # Prepare for pyg_data.Data
    adjacency_list = None

    graph = MOTGraph(
        edge_index=adjacency_list,
        gt_adjacency_dict=gt_dict,
        y=None,
        detections=detections,
        detections_coords=detections_coords,
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

        # knn graph is somehow not respecting the original edges,
        if mps_fallback:
            graph = graph.to('mps')

    # remove duplicates and make graph true undirected
    sources, targets = graph.edge_index
    mask = sources < targets
    graph.edge_index = graph.edge_index[:, mask]

    # linkage

    # time mask
    sources, targets = graph.edge_index
    time_distances = (frame_times[targets] - frame_times[sources]).squeeze()

    if linkage_window == -1:
        linkage_window = torch.inf
    elif linkage_window == 0:
        linkage_window = 1
    mask = (time_distances < linkage_window) & (time_distances != 0)
    graph.edge_index = graph.edge_index[:, mask]

    graph.edge_index = shuffle_tensor(graph.edge_index.t()).t()
    # assert knn didn't delete gt
    # gt_adjacency_set = set([tuple(x) for x in gt_dict['1']])
    # # # assert no ground truth has been lost
    # test_list = graph.edge_index.t().tolist()
    # test = [list(a) in test_list for a in gt_adjacency_set]
    # # assert all(a is True for a in test)
    # del test, test_list


    # # once the graph is pruned, compute edge attributes
    edge_attributes = torch.zeros(graph.edge_index.shape[1], EDGE_FEATURES_DIM).to(device)

    # Extract source and target nodes for each edge
    sources, targets = graph.edge_index

    # avoid infinity with EPSILON

    # Compute x, y, h, w, and t attributes for each edge using tensor operations
    edge_attributes[:, 0] = (2 * (detections_coords[targets, 0] - detections_coords[sources, 0])) / ((
            detections_coords[sources, 2] + detections_coords[targets, 2]) + EPSILON)
    edge_attributes[:, 1] = (2 * (detections_coords[sources, 1] - detections_coords[targets, 1])) / (
        (detections_coords[sources, 3] + detections_coords[targets, 3]) + EPSILON)
    edge_attributes[:, 2] = torch.log(detections_coords[sources, 2] / (detections_coords[targets, 2]) + EPSILON)
    edge_attributes[:, 3] = torch.log(detections_coords[sources, 3] / (detections_coords[targets, 3]) + EPSILON)
    edge_attributes[:, 4] = (frame_times[targets] - frame_times[sources]).squeeze() / (frame_times[-1] + EPSILON)
    edge_attributes[:, 5] = 1 / (distance_matrix[sources, targets] + EPSILON)

    graph.edge_attr = edge_attributes

    # Build `y` tensor to compare predictions with gt
    weight_potencies = list(gt_dict.keys())

    if gt_dict is not None:
        y = torch.zeros(len(graph.edge_attr)).to(device=device, dtype=dtype)
        for i, x in enumerate(graph.edge_index.t().tolist()):
            for potency in weight_potencies:
                gt_list = gt_dict[potency]
                if x in gt_list:
                    y[i] = 1 / int(potency)
                    break
        graph.y = y

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
                 classification=False,
                 backbone:str = 'resnet50'):

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
        self.classification = classification
        self.backbone = ImgEncoder(model_name=backbone, dtype=dtype).to(device=device)

        logging.info(f"{self.n_frames} frames")

        # Check if the chosen linkage window is possible
        if self.linkage_window > self.n_frames:
            logging.warning(f"`linkage window` was set to {self.linkage_window} but track has {self.n_frames} frames."
                            f"Setting `linkage window` to {self.n_frames}")
            self.linkage_window = self.n_frames

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

        number_of_detections = sum([len(x) for x in self.detections])

        track_detections_coords = torch.zeros((number_of_detections, 4), dtype=self.dtype).to(self.device)
        frame_times = torch.zeros((number_of_detections, 1), dtype=torch.int16).to(self.device)
        image_container = torch.zeros((number_of_detections, channels, self.det_resize[1], self.det_resize[0]),
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

                image_container[j, :, :, :] = detection

                track_detections_coords[j, :] = torch.tensor(bbox)
                frame_times[j, 0] = i
                j += 1

            # List with the number of nodes per frame (aka number of detections per frame)
            self.n_nodes.append(nodes)
            i += 1
        with torch.no_grad():
            node_features = self.backbone(image_container)


        """
        Node linkage - ground truth
        """

        gt_adjacency_list = None

        n_sum = [0] + np.cumsum(self.n_nodes).tolist()
        # If detections ids are available (!= -1)
        if self.detections[0][0]['id'] != -1:

            gt_adjacency_list = list()

            # Get list of all gt detections ids
            gt_detections_ids = list(set([detection['id'] for frame in self.detections for detection in frame]))

            all_paths = list()

            # Iterate over all the detections ids (aka over all gt trajectories)
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

            # build ground truth dict

            gt_dict = dict()


            if self.classification:
                gt_dict['1'] = []
                for path in all_paths:
                    for j in range(len(path) - 1):
                        try:
                            gt_dict['1'].append([path[j], path[j + 1]])
                            # gt_adjacency_list.append([path[j + 1], path[j]]) # -------------------------------------------------
                        except:
                            continue
            else:
                for i in range(1, self.linkage_window + 1):
                    gt_dict[str(i)] = []
                # Fill gt_adjacency_list using the detections in all_paths
                for i in range(1, self.linkage_window + 1):
                    for path in all_paths:
                        for j in range(len(path) - 1):
                            try:
                                gt_dict[str(i)].append([path[j], path[j + i]])
                                # gt_adjacency_list.append([path[i + 1], path[i]]) # -------------------------------------------------
                            except:
                                continue

            logging.info(f"{len(gt_adjacency_list)} total gt edges ({len(all_paths)} trajectories)")

        else:
            logging.info(f"No ground truth adjacency list available")
            gt_dict = None

        """
        Output section
        """

        return {
            # "adjacency_list": adjacency_list,
            'n_nodes':self.n_nodes,
            'linkage_window':self.linkage_window,
            "gt_dict": gt_dict,
            "detections": node_features,
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
                 linkage_window: int = 5,
                 subtrack_len: int = 15,
                 slide: int = 10,
                 dl_mode: bool = True,
                 black_and_white_features=False,
                 naive_pruning_args=None,
                 knn_pruning_args=None,
                 mps_fallback: bool = False,
                 device: torch.device = torch.device("cuda"),
                 dtype=torch.float32,
                 classification=False,
                 preprocessing: bool = False,
                 preprocessed: bool = True,
                 preprocessed_data_folder: str = 'preprocessed_data',
                 feature_extraction_backbone:str = "resnet50"
                 ):
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
        self.classification = classification
        self.preprocessed = preprocessed
        self.preprocessed_data_folder = preprocessed_data_folder
        self.preprocessing = preprocessing
        self.backbone = feature_extraction_backbone

        if self.preprocessing:
            print("[INFO] Data loader set in preprocessing mode")
        if self.preprocessed:
            print("[INFO] Data loader set in preprocessed mode")
        if self.preprocessing and self.preprocessed:
            raise Exception("Data loader cannot be in both preprocessing and preprocessed mode")

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

    def _build_preprocess_path(self, idx) -> str:
        operation = "classification" if self.classification else "regression"
        path = os.path.normpath(
            os.path.join(
                self.preprocessed_data_folder, operation, self.name,self.backbone,
                self.tracklist[self.cur_track]
            )
        )
        create_folders(path)
        return os.path.normpath(
            os.path.join(
                path, str(idx) + ".pkl"
            )
        )

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
            for start_frame in range(0, n_frames - self.slide + 1, self.slide):
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

        # Get the starting frame and track for this batch
        starting_frame = self.start_frames[idx]
        cur_track = self.tracks[idx]
        ending_frame = starting_frame + self.subtrack_len

        self.cur_track = cur_track
        self.str_frame = starting_frame
        self.end_frame = ending_frame

        if self.preprocessed:
            load_path = self._build_preprocess_path(idx)
            with open(load_path, 'rb') as f:
                tracklet_graph = pickle.load(f)
            return tracklet_graph

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

        if self.frames_per_track[cur_track] - starting_frame < 15:
            ending_frame = starting_frame + (self.frames_per_track[cur_track] - starting_frame)

        logging.debug(f"From {starting_frame} to {ending_frame}")
        logging.debug(f"Starting in track: {cur_track}")
        logging.debug(f"From {starting_frame} to {ending_frame} of track {cur_track}")

        frames_window_msg = f"frames {starting_frame}-{ending_frame}/{len(all_images[cur_track])}"

        logging.info(
            f"Subtrack #{idx} | track {self.tracklist[cur_track]} {frames_window_msg}\r")

        track = MotTrack(detections=all_detections[cur_track][starting_frame:ending_frame],
                         images_list=all_images[cur_track][starting_frame:ending_frame],
                         det_resize=self.det_resize,
                         linkage_window=self.linkage_window,
                         subtrack_len=self.subtrack_len,
                         classification=self.classification,
                         device=self.device,
                         dtype=self.dtype,
                         logging_lv=logging.WARNING if self.dl_mode else logging.INFO,
                         name=self.name + "/track_" + str(self.tracklist[cur_track]) + "/subtrack_" + str(idx),
                         backbone=self.backbone)

        if self.dl_mode:
            graph = build_graph(
                mps_fallback=self.mps_fallback,
                device=self.device,
                dtype=self.dtype,
                knn_pruning_args=self.knn_pruning_args,
                **track.get_data())
            if self.preprocessing:
                save_path = self._build_preprocess_path(idx)
                with open(save_path, "wb") as f:
                    pickle.dump(graph, f)
                    return None
            else:
                return graph
        else:
            return track
