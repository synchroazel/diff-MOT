import collections
import os

from PIL import Image
from torch.utils.data import Dataset


class MotTrack:

    def __init__(self, track_path):
        self.track_dir = track_path

        self.det_file = os.path.join(self.track_dir, "det", "det.txt")
        self.img_dir = os.path.join(self.track_dir, "img1")

        self.detections = self._read_detections()

    def _read_detections(self):
        detections = collections.defaultdict(list)
        with open(self.det_file, "r") as f:
            for line in f.readlines():
                line = line.split(",")
                det_bbox = [float(el) for el in line[2:6]]
                detections[int(line[0])].append(det_bbox)
        return detections

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.img_dir, f"{item + 1:06d}.jpg"))
        detections = self.detections[item + 1]
        return image, detections


class MotDataset(Dataset):

    def __init__(self, dataset_path, split):
        self.dataset_dir = dataset_path
        self.split = split

        assert split in os.listdir(self.dataset_dir), \
            f"Split must be one of {os.listdir(self.dataset_dir)}."

        self.tracklist = os.listdir(os.path.join(self.dataset_dir, split))

    def __getitem__(self, item):
        return MotTrack(os.path.join(self.dataset_dir, self.split, self.tracklist[item]))
