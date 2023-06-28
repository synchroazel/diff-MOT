import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms

from utilities import get_best_device

class ImgEncoder:
    """
    Image encoder class.
    Takes in an image and model name from torchvision.models and returns a feature vector.
    """

    def __init__(self, model_name):
        self.device = get_best_device()
        self.model = models.__dict__[model_name]()
        self.model.fc = nn.Identity()
        self.model.to(self.device)

    def __call__(self, img):
        self.model.eval()

        img_ = transforms.ToTensor()(img)
        img_ = Variable(img_.unsqueeze(0))

        with torch.no_grad():
            feats = self.model(img_.to(self.device))
            feats = feats.squeeze(0)

        return feats
