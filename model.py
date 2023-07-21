import torch
import torch.nn.functional as F
import torch_geometric.nn
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torchvision import models


class ImgEncoder(torch.nn.Module):
    def __init__(self, model_name: str, weights: str = "DEFAULT", dtype=torch.float32):
        super(ImgEncoder, self).__init__()

        self.model_name = model_name.lower()
        self.weights = 'DEFAULT'

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

        self.model.eval()
        self.model.to(dtype=dtype)

    def forward(self, img):
        with torch.no_grad():
            features = self.model(img)
            features = features.reshape(features.size(0), -1)
            return features


class EdgePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgePredictor, self).__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x_i, x_j):
        # x_i and x_j have shape [E, in_channels]
        x = torch.cat([x_i, x_j], dim=-1)  # Concatenate node features.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x).squeeze()
        # x = torch.sigmoid(x).squeeze()
        return x


class Net(torch.nn.Module):
    # We should use only layers that support edge features
    layer_aliases = {
        'GATConv': torch_geometric.nn.GATConv,  # https://arxiv.org/abs/1710.10903
        'GATv2Conv': torch_geometric.nn.GATv2Conv,  # https://arxiv.org/abs/2105.14491
        'TransformerConv': torch_geometric.nn.TransformerConv,  # https://arxiv.org/abs/2009.03509
        # 'GMMConv': torch_geometric.nn.GMMConv,  # https://arxiv.org/abs/1611.08402
        # 'SplineConv': torch_geometric.nn.SplineConv,  # https://arxiv.org/abs/1711.08920
        # 'NNConv': torch_geometric.nn.NNConv, # https://arxiv.org/abs/1704.01212
        # 'CGConv': torch_geometric.nn.CGConv,  # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
        # 'PNAConv': torch_geometric.nn.PNAConv,  # https://arxiv.org/abs/2004.05718
        'GENConv': torch_geometric.nn.GENConv,  # https://arxiv.org/abs/2006.07739
        'GeneralConv': torch_geometric.nn.GeneralConv,  # https://arxiv.org/abs/2011.08843
    }

    def __init__(self, backbone, layer_size, layer_tipe='GATConv', dtype=torch.float32, mps_fallback=False, **kwargs):
        super(Net, self).__init__()
        self.fextractor = ImgEncoder(backbone, dtype=dtype)
        self.layer_type = layer_tipe
        self.layer_size = layer_size
        self.backbone = backbone
        self.conv1 = self.layer_aliases[layer_tipe](in_channels=-1, out_channels=layer_size, **kwargs)
        self.conv2 = self.layer_aliases[layer_tipe](in_channels=-1, out_channels=layer_size, **kwargs)
        self.predictor = EdgePredictor(layer_size, layer_size)
        self.dtype = dtype
        self.device = next(self.parameters()).device  # get the device the model is currently on
        self.mps_fallback = mps_fallback
        self.to(dtype=dtype)

        # Fallback to CPU if device is MPS
        if self.mps_fallback:
            print('[INFO] Falling back to CPU for conv layers.')

    def forward(self, data):
        data.x = self.fextractor(data.detections)

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        del data

        # Fallback to CPU if device is MPS
        if self.mps_fallback:
            x = x.to(torch.device('cpu'))
            edge_index = edge_index.to(torch.device('cpu'))
            edge_attr = edge_attr.to(torch.device('cpu'))
            self.conv1.to('cpu')
            self.conv2.to('cpu')


        # todo: edge features updates? we can update the features after the node update. Double check lin_edge inside conv
        x = self.conv1(x=x, edge_index=edge_index.to(torch.int64), edge_attr=edge_attr)
        # x = F.relu(x)  # some layers already have activation, but not all of them
        # x = F.dropout(x, training=self.training, p=0.2)  # not all layers have incorporated dropout, so we put it here
        x = self.conv2(x=x, edge_index=edge_index.to(torch.int64), edge_attr=edge_attr)

        # Back on MPS after the convolutions
        if self.mps_fallback:
            x = x.to(torch.device('mps'))

        # Use edge_index to find the corresponding node features in x
        x_i, x_j = x[edge_index[0].to(torch.int64)], x[edge_index[1].to(torch.int64)]

        return self.predictor(x_i, x_j)

    def __str__(self):
        return self.layer_type.lower() + "_" + str(self.layer_size) + "_" + self.backbone + "-backbone"
