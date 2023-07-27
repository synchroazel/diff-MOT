import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_geometric.nn
from efficientnet_pytorch import EfficientNet
from torch import Tensor
from torch import nn
from torch_geometric.typing import OptTensor, PairTensor, SparseTensor, Adj
from torch_geometric.utils import softmax
from torchvision import models

# todo: capire come gestire la variabilità dei layers

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
        x = self.lin2(x)#.squeeze()
        x = torch.sigmoid(x).squeeze()
        return x

class TransformerConvWithEdgeUpdate(torch_geometric.nn.TransformerConv):
    def __init__(self,**kwargs):
        super(TransformerConvWithEdgeUpdate, self).__init__(**kwargs)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # same code as og class

        if self.lin_edge is not None:
            assert edge_attr is not None
            # ----> edges are updeted here <-----
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        # out = torch.mean(out, dim=1)
        out = out * alpha.view(-1, self.heads, 1)

        #  ----> save edge attr <-----
        self.__edge_attr__ = torch.mean(edge_attr, 1) # TODO: should we put key instead?

        return out

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        # same code as og class
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            # ------> MODIFIED HERE <---------
            edge_features = self.__edge_attr__
            self.__edge_attr__ = None
            return out, edge_features

class Net(torch.nn.Module):
    # We should use only layers that support edge features
    layer_aliases = {
        'GATConv': torch_geometric.nn.GATConv,  # https://arxiv.org/abs/1710.10903
        'GATv2Conv': torch_geometric.nn.GATv2Conv,  # https://arxiv.org/abs/2105.14491
        'TransformerConv': TransformerConvWithEdgeUpdate,  # https://arxiv.org/abs/2009.03509
        # 'GMMConv': torch_geometric.nn.GMMConv,  # https://arxiv.org/abs/1611.08402
        # 'SplineConv': torch_geometric.nn.SplineConv,  # https://arxiv.org/abs/1711.08920
        # 'NNConv': torch_geometric.nn.NNConv, # https://arxiv.org/abs/1704.01212
        # 'CGConv': torch_geometric.nn.CGConv,  # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
        # 'PNAConv': torch_geometric.nn.PNAConv,  # https://arxiv.org/abs/2004.05718
        'GENConv': torch_geometric.nn.GENConv,  # https://arxiv.org/abs/2006.07739
        'GeneralConv': torch_geometric.nn.GeneralConv,  # https://arxiv.org/abs/2011.08843
    }

    def __init__(self, backbone, layer_size,steps=2, layer_tipe='TransformerConv', dtype=torch.float32, mps_fallback=False, **kwargs):
        super(Net, self).__init__()
        self.fextractor = ImgEncoder(backbone, dtype=dtype)
        self.layer_type = layer_tipe
        self.layer_size = layer_size
        self.backbone = backbone
        self.conv1 = self.layer_aliases[layer_tipe](in_channels=-1, out_channels=layer_size, **kwargs)
        kwargs['edge_dim'] = layer_size
        for i in range(2, steps+1):
            layer = "self.conv" + str(i) +" = self.layer_aliases[layer_tipe](in_channels=-1, out_channels=layer_size,**kwargs)"
            exec( layer)
        self.number_of_message_passing_layers = steps
        self.predictor = EdgePredictor(layer_size, layer_size)
        self.dtype = dtype
        self.device = next(self.parameters()).device  # get the device the model is currently on
        self.mps_fallback = mps_fallback
        self.to(dtype=dtype)

        # Fallback to CPU if device is MPS
        if self.mps_fallback:
            print('[INFO] Falling back to CPU for conv layers.')

    def forward(self, data)-> tuple:
        # 1
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

        # 2
        for i in range(1, self.number_of_message_passing_layers + 1):
            message_passing = " self.conv"+str(i)+"(x=x, edge_index=edge_index.to(torch.int64), edge_attr=edge_attr)"
            x, edge_attr = eval(message_passing)

        # Back on MPS after the convolutions
        if self.mps_fallback:
            x = x.to(torch.device('mps'))

        # Use edge_index to find the corresponding node features in x
        x_i, x_j = x[edge_index[0].to(torch.int64)], x[edge_index[1].to(torch.int64)]

        return self.predictor(x_i, x_j)# , edge_attr

    def __str__(self):
        return self.layer_type.lower() + "_" + str(self.layer_size) + "_" + self.backbone + "-backbone"
