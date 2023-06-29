import torch


def get_best_device():
	"""Function used to get the best device between cuda, mps and cpu"""
	if torch.cuda.is_available():
		print("[INFO] Using CUDA.")
		return torch.device("cuda")
	elif torch.has_mps:
		print("[INFO] Using MPS.")
		return torch.device("mps")
	else:
		print("[INFO] No GPU found. Using CPU.")
		return torch.device("cpu")
