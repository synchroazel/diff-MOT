import collections
import os
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric.data as pyg_data
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_convert

from encoders import ImgEncoder


class MotTrack:

	def __init__(self, track_path):
		self.track_dir = track_path
		self.det_file = os.path.join(self.track_dir, "det", "det.txt")
		self.img_dir = os.path.join(self.track_dir, "img1")
		self.detections = self._read_detections()
		self.img_encoder = ImgEncoder("resnet50")

	def _read_detections(self):

		file = np.loadtxt(self.det_file, delimiter=",", usecols=(0, 2, 3, 4, 5))

		dets = {}
		for det in file:
			frame = int(det[0])
			if frame not in dets:
				dets[frame] = []
			dets[frame].append(det[1:].tolist())

		return dets

	def get_graph(self, limit=None):

		track_enc_dets = []  # all encoded detections across all frames

		pbar = tqdm(
			sorted(os.listdir(self.img_dir))
		)

		i = 0  # frame counter

		for image in pbar:

			pbar.set_description(f"Processing frame #{i} ({image})")

			image = Image.open(os.path.join(self.img_dir, image))

			frame_enc_dets = []  # all encoded detections in the current frame

			for j, bbox in enumerate(self.detections[i + 1]):
				bbox = box_convert(torch.tensor(bbox), "xywh", "xyxy").tolist()

				det_image = image.crop(bbox)

				frame_enc_dets.append(
					self.img_encoder(det_image)
				)

			# print(f"[INFO] {len(frame_enc_dets)} encoded detections in frame #{i}")

			track_enc_dets.append(frame_enc_dets)

			if i + 1 == limit:
				break

			i += 1

		n_nodes = [len(frame_dets) for frame_dets in track_enc_dets]
		n_sum = [0] + np.cumsum(n_nodes).tolist()
		edge_index = []
		for i in range(len(track_enc_dets) - 1):
			for j in range(len(track_enc_dets[i])):
				for k in range(len(track_enc_dets[i + 1])):
					edge_index.append([
						j + n_sum[i],
						k + np.cumsum(n_nodes[:i + 1])[-1]
					])
					edge_index.append([
						k + np.cumsum(n_nodes[:i + 1])[-1],
						j + n_sum[i]
					])

		print(f"[INFO] {len(edge_index)} total edges")

		track_enc_dets = [item.cpu().numpy()
						  for sublist in track_enc_dets
						  for item in sublist]

		# Create a pytorch geometric data object

		graph = pyg_data.Data(
			x=torch.tensor(track_enc_dets),
			edge_index=torch.tensor(edge_index).t().contiguous()
		)

		return graph, n_nodes


class MotDataset(Dataset):

	def __init__(self, dataset_path, split):
		self.dataset_dir = dataset_path
		self.split = split

		assert split in os.listdir(self.dataset_dir), \
			f"Split must be one of {os.listdir(self.dataset_dir)}."

		self.tracklist = os.listdir(os.path.join(self.dataset_dir, split))

	def __getitem__(self, item):
		return MotTrack(os.path.join(self.dataset_dir, self.split, self.tracklist[item]))
