"""This file contains the class ImgEncoder, which will be used to encode an image into a set of features"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms


class ImgEncoder:
    """
    Image encoder class.
    Takes in an image and model name from torchvision.models and returns a feature vector.
    """

    def __init__(self, model_name: str, device: str):
        self.model = models.__dict__[model_name]()
        self.model.fc = nn.Identity()
        self.model.to(device)
        self.device = device

    def __call__(self, img):
        self.model.eval()

        img_tensor = transforms.ToTensor()(img).unsqueeze(0)

        with torch.no_grad():
            features = self.model(img_tensor.to(self.device))
            features = features.squeeze(0)

        return features
