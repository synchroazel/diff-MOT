import logging
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_geometric.nn
from efficientnet_pytorch import EfficientNet
from torch import Tensor
from torch import nn
from torch_geometric.nn.dense.linear import Linear, Parameter
from torch_geometric.typing import OptTensor, PairTensor, SparseTensor, Adj
from torch_geometric.utils import add_self_loops, degree, softmax
from torchvision import models
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, Optional, OptPairTensor, OptTensor, Size
from torch_scatter import (
    scatter_sum,
    scatter_add,
    scatter_mul,
    scatter_mean,
    scatter_min,
    scatter_max,
    scatter_std,
    scatter_logsumexp,
    scatter_softmax,
    scatter_log_softmax
)


# TODO: capire come gestire la variabilità dei layers

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


class EdgePredictorFromEdges(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, edge_attr):
        x = F.relu(self.lin1(edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        # x = torch.sigmoid(x)  # CLARIFY SIGMOID THING
        x = x.squeeze()
        return x


class EdgePredictorFromNodes(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(4362, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x_i, x_j):
        # x_i and x_j have shape [E, in_channels]
        x = torch.cat([x_i, x_j], dim=-1)  # Concatenate node features.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x).squeeze()
        # x = torch.sigmoid(x).squeeze()
        return x


class TransformerConvWithEdgeUpdate(torch_geometric.nn.TransformerConv):
    def __init__(self, **kwargs):
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
        self.__edge_attr__ = torch.mean(edge_attr, 1)  # TODO: should we put key instead?

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


class EdgeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals):
        super().__init__()
        self.residuals = residuals
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class TimeAwareNodeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals, agg_future, agg_past):
        super(TimeAwareNodeModel, self).__init__()
        self.residuals = residuals
        self.node_mlp_future = nn.Sequential(
            nn.Linear(n_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )
        self.node_mlp_past = nn.Sequential(
            nn.Linear(n_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )
        self.node_mlp_combine = nn.Sequential(
            nn.Linear(n_targets * 2, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

        self.agg_future = agg_future
        self.agg_past = agg_past
        self.agg_function = {
            'sum': scatter_sum,
            'add': scatter_add,
            'mul': scatter_mul,
            'mean': scatter_mean,
            'min': scatter_min,
            'max': scatter_max,
            'std': scatter_std,
            'logsumexp': scatter_logsumexp,
            'softmax': scatter_softmax,
            'log_softmax': scatter_log_softmax,
        }

    def forward(self, x, edge_index, edge_attr, u, batch):
        n1, n2 = edge_index

        future_mask = n1 < n2
        future_n1, future_n2 = n1[future_mask], n2[future_mask]
        future_in = torch.cat([x[future_n2], edge_attr[future_mask]], dim=1)
        future_out = self.node_mlp_future(future_in)
        future_out = self.agg_function[self.agg_future](future_out, future_n1, dim=0, dim_size=x.size(0))

        past_mask = n1 > n2
        past_n1, past_n2 = n1[past_mask], n2[past_mask]
        past_in = torch.cat([x[past_n2], edge_attr[past_mask]], dim=1)
        past_out = self.node_mlp_past(past_in)
        past_out = self.agg_function[self.agg_past](past_out, past_n1, dim=0, dim_size=x.size(0))

        flow = torch.cat((past_out, future_out), dim=1)

        out = self.node_mlp_combine(flow)

        if self.residuals:
            out = out + x
        return out


def build_custom_mp(n_target_nodes, n_target_edges, n_features, n_edge_features, layer_size, residuals, device="cuda"):
    return torch_geometric.nn.MetaLayer(
        edge_model=EdgeModel(n_features=n_features, n_edge_features=n_edge_features, hiddens=layer_size,
                             n_targets=n_target_edges, residuals=residuals),
        node_model=TimeAwareNodeModel(n_features=n_features, n_edge_features=n_target_edges, hiddens=layer_size,
                                      n_targets=n_target_nodes, residuals=residuals,
                                      agg_future='sum', agg_past='mean')
    ).to(device=device)


class GeneralConvWithEdgeUpdate(torch_geometric.nn.MessagePassing):
    r"""A general GNN layer adapted from the `"Design Space for Graph Neural
    Networks" <https://arxiv.org/abs/2011.08843>`_ paper."""

    def __init__(
            self,
            in_channels: Union[int, tuple[int, int]],
            out_channels: Optional[int],
            in_edge_channels: int = None,
            aggr: str = "add",
            skip_linear: str = False,
            directed_msg: bool = True,
            heads: int = 1,
            attention: bool = False,
            attention_type: str = "additive",
            l2_normalize: bool = False,
            bias: bool = True,
            device="cuda",
            **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edge_channels = in_edge_channels
        self.aggr = aggr
        self.skip_linear = skip_linear
        self.directed_msg = directed_msg
        self.heads = heads
        self.attention = attention
        self.attention_type = attention_type
        self.normalize_l2 = l2_normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.directed_msg:
            self.lin_msg = Linear(in_channels[0], out_channels * self.heads,
                                  bias=bias).to(device)
        else:
            self.lin_msg = Linear(in_channels[0], out_channels * self.heads,
                                  bias=bias).to(device)
            self.lin_msg_i = Linear(in_channels[0], out_channels * self.heads,
                                    bias=bias).to(device)

        if self.skip_linear or self.in_channels != self.out_channels:
            self.lin_self = Linear(in_channels[1], out_channels, bias=bias)
        else:
            self.lin_self = torch.nn.Identity()

        if self.in_edge_channels is not None:
            self.lin_edge = Linear(in_edge_channels, out_channels * self.heads,
                                   bias=bias)

        # TODO: A general torch_geometric.nn.AttentionLayer
        if self.attention:
            if self.attention_type == 'additive':
                self.att_msg = Parameter(
                    torch.Tensor(1, self.heads, self.out_channels))
            elif self.attention_type == 'dot_product':
                self.scaler = torch.sqrt(
                    torch.tensor(out_channels, dtype=torch.float))
            else:
                raise ValueError(
                    f"Attention type '{self.attention_type}' not supported")

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_msg.reset_parameters()
        if hasattr(self.lin_self, 'reset_parameters'):
            self.lin_self.reset_parameters()
        if self.in_edge_channels is not None:
            self.lin_edge.reset_parameters()
        if self.attention and self.attention_type == 'additive':
            glorot(self.att_msg)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Tensor = None, size: Size = None) -> tuple[Tensor, Tensor]:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_self = x[1]
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        out = out.mean(dim=1)  # todo: other approach to aggregate heads
        out = out + self.lin_self(x_self)
        if self.normalize_l2:
            out = F.normalize(out, p=2, dim=-1)

        #  ----> save edge attr <-----
        edge_attr = self.__edge_attr__
        self.__edge_attr__ = None

        return out, edge_attr

    def message_basic(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor):
        if self.directed_msg:
            x_j = self.lin_msg(x_j)
        else:
            x_j = self.lin_msg(x_j) + self.lin_msg_i(x_i)
        if edge_attr is not None:
            lin_edge_attr = self.lin_edge(edge_attr)
            x_j = x_j + lin_edge_attr
            self.__edge_attr__ = lin_edge_attr
        return x_j

    def message(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor,
                size_i: Tensor, edge_attr: Tensor) -> Tensor:
        x_j_out = self.message_basic(x_i, x_j, edge_attr)
        x_j_out = x_j_out.view(-1, self.heads, self.out_channels)
        if self.attention:
            if self.attention_type == 'dot_product':
                x_i_out = self.message_basic(x_j, x_i, edge_attr)
                x_i_out = x_i_out.view(-1, self.heads, self.out_channels)
                alpha = (x_i_out * x_j_out).sum(dim=-1) / self.scaler
            else:
                alpha = (x_j_out * self.att_msg).sum(dim=-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            alpha = alpha.view(-1, self.heads, 1)
            return x_j_out * alpha
        else:
            return x_j_out


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
        'GeneralConv': GeneralConvWithEdgeUpdate  # https://arxiv.org/abs/2011.08843
    }

    def __init__(self,
                 backbone,
                 layer_size=256,
                 n_target_nodes=256,
                 n_target_edges=256,
                 steps=2,
                 edge_features_dim=2,
                 residuals=True,
                 layer_tipe='TimeAware',
                 dtype=torch.float32,
                 mps_fallback=False,
                 device='cuda:0',
                 **kwargs):
        super(Net, self).__init__()
        self.fextractor = ImgEncoder(backbone, dtype=dtype)
        self.layer_type = layer_tipe
        self.layer_size = layer_size
        self.backbone = backbone

        # self.conv_in = self.layer_aliases[layer_tipe](in_channels=-1, out_channels=layer_size, **kwargs)
        # tODO: n_features
        self.conv_in = build_custom_mp(n_target_nodes=n_target_nodes, n_target_edges=n_target_edges, n_features=2048, n_edge_features=edge_features_dim,
                                       layer_size=layer_size, residuals=False, device=device)

        kwargs['edge_dim'] = layer_size
        kwargs['in_edge_channels'] = layer_size * kwargs['heads']

        # for i in range(2, steps + 1):
        #     layer = "self.conv" + str(
        #         i) + " = self.layer_aliases[layer_tipe](in_channels=-1, out_channels=layer_size,**kwargs)"
        #     exec(layer)

        self.number_of_message_passing_layers = steps

        self.conv = []
        for i in range(steps - 1):
            # self.conv.append(self.layer_aliases[layer_tipe](in_channels=-1, out_channels=layer_size, **kwargs))
            self.conv.append(
                build_custom_mp(n_target_nodes=n_target_nodes, n_target_edges=n_target_edges, n_features=n_target_nodes, n_edge_features=n_target_edges,
                                       layer_size=layer_size, residuals=residuals, device=device)
            )

        # self.predictor = EdgePredictor(layer_size, layer_size)
        self.predictor = EdgePredictorFromEdges(in_channels=n_target_edges, hidden_channels=128)  # ??? HELP HERE
        self.dtype = dtype
        self.device = device  # get the device the model is currently on
        self.mps_fallback = mps_fallback
        self.to(dtype=dtype)

        # Fallback to CPU if device is MPS
        if self.mps_fallback:
            print('[INFO] Falling back to CPU for conv layers.')

    def forward(self, data) -> tuple:

        # Step 1 - Extract features from the image

        data.x = self.fextractor(data.detections)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        del data

        # Step 2 - Enter the Message Passing layers

        # Fallback to CPU if device is MPS
        if self.mps_fallback:
            x = x.to(torch.device('cpu'))
            edge_index = edge_index.to(torch.device('cpu'))
            edge_attr = edge_attr.to(torch.device('cpu'))
            self.conv_in.to(torch.device('cpu'))
            for layer in self.conv:
                layer.to(torch.device('cpu'))

        # 2
        # for i in range(1, self.number_of_message_passing_layers + 1):
        #     message_passing = " self.conv" + str(
        #         i) + "(x=x, edge_index=edge_index.to(torch.int64), edge_attr=edge_attr)"
        #     x, edge_attr = eval(message_passing)

        x, edge_attr, _ = self.conv_in(x=x, edge_index=edge_index.to(torch.int64), edge_attr=edge_attr)

        for i in range(self.number_of_message_passing_layers - 1):
            x, edge_attr, _ = self.conv[i](x=x, edge_index=edge_index.to(torch.int64), edge_attr=edge_attr)

        # Back on MPS after the convolutions
        if self.mps_fallback:
            x = x.to(torch.device('mps'))
            edge_attr = edge_attr.to(torch.device('mps'))

        # Use edge_index to find the corresponding node features in x
        x_i, x_j = x[edge_index[0].to(torch.int64)], x[edge_index[1].to(torch.int64)]

        return self.predictor(edge_attr)

        # return self.predictor(x_i, x_j)

    def __str__(self):
        return self.layer_type.lower() + "_" + str(self.layer_size) + "_" + self.backbone + "-backbone"
