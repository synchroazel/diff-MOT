import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torchvision import models


class ImgEncoder(torch.nn.Module):
    def __init__(self, model_name: str, weights: str = "DEFAULT", dtype = torch.float32):
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


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 5: Normalize node features.
        return norm.view(-1, 1) * x_j


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
        x = torch.sigmoid(self.lin2(x)).squeeze()
        return x


class Net(torch.nn.Module):

    def __init__(self, backbone, layer_size, dtype = torch.float32):
        super(Net, self).__init__()
        self.fextractor = ImgEncoder(backbone, dtype = dtype)
        self.conv1 = GCNConv(2048, layer_size)  # TODO: dynamic n_features
        self.conv2 = GCNConv(layer_size, layer_size)
        self.predictor = EdgePredictor(layer_size, layer_size)
        self.dtype = dtype
        self.to(dtype=dtype)
    def forward(self, data):
        data.x = self.fextractor(data.detections)

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x=x, edge_index=edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index)

        # Use edge_index to find the corresponding node features in x
        x_i, x_j = x[edge_index[0].to(torch.long)], x[edge_index[1].to(torch.long)]

        return self.predictor(x_i, x_j)
