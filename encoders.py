"""This file contains the class ImgEncoder, which will be used to encode an image into a set of features"""
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import models


class ImgEncoder:
    """
    Image encoder class.
    Takes in an image and model name from torchvision.models and returns a feature vector.
    """

    def __init__(self, model_name: str, weights: str = "DEFAULT", device: str = "cpu"):
        self.model_name = model_name.lower()
        self.weights = 'DEFAULT'
        self.device = device

        # Load the chosen model
        if 'resnet' in self.model_name:
            self.model = getattr(models, self.model_name)(weights=weights)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        elif 'vgg' in self.model_name:
            self.model = getattr(models, self.model_name)(weights=weights)
            self.model.classifier = self.model.classifier[:-1]

        elif 'vit' in self.model_name:
            self.model = getattr(models, self.model_name)(weights=weights)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        elif 'efficientnet' in self.model_name:
            self.model = EfficientNet.from_pretrained(self.model_name)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        else:
            raise ValueError('Invalid model name.')

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, img):
        with torch.no_grad():
            features = self.model(img.to(self.device))
            features = features.reshape(features.size(0), -1)
            return features
