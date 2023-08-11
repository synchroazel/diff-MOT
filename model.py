import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch_geometric.nn
from efficientnet_pytorch import EfficientNet
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
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
from torchvision import models
from  utilities import *

# TODO: capire come gestire la variabilità dei layers


class ImgEncoder(torch.nn.Module):
    output_dims = {
        'resnet50': 2048,
        'resnet101': 2048,
        'vit_l_32': 1024,
        'vgg16': 4096,
        'vgg19': 4096,
        # 'efficientnet-b0': 1000,
        # 'efficientnet-b7':2560
    }

    def __init__(self, model_name: str, weights: str = "DEFAULT", dtype=torch.float32):
        super(ImgEncoder, self).__init__()

        self.model_name = model_name.lower()
        self.weights = 'DEFAULT'
        self.output_dim = self.output_dims[model_name]

        # TODO: set outdim
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(out_channels, 1)

    def forward(self, edge_attr):
        x = F.leaky_relu(self.lin1(edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = x.squeeze()
        return x


class EdgePredictorFromNodes(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, out_channels)
        self.lin2 = nn.Linear(out_channels, 1)

    def forward(self, x_i, x_j):
        # x_i and x_j have shape [E, in_channels]
        x = torch.cat([x_i, x_j], dim=-1)  # Concatenate node features.
        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x).squeeze()
        return x

# ok
class TransformerConvWithEdgeUpdate(torch_geometric.nn.TransformerConv):
    def __init__(self,
                 edge_model: bool = False,
                 in_channels:int = 500, out_channels:int = 500,
                 heads:int=1, dropout:float=.3,
                 **padding_kwargs):
        super(TransformerConvWithEdgeUpdate, self).__init__(in_channels=in_channels, out_channels=out_channels,heads=heads,
                                                            dropout=dropout)
        self.edge_model = edge_model

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # same code as og class

        if self.lin_edge is not None:
            #     assert edge_attr is not None
            #     # ----> edges are updeted here <-----
            #     edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
            #                                               self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            try:
                out += edge_attr
            except:
                pass

        # out = torch.mean(out, dim=1)
        out = out * alpha.view(-1, self.heads, 1)

        #  ----> save edge attr <-----
        self.__edge_attr__ = torch.mean(key_j, 1)

        return out

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr = None, u=None, batch=None, return_attention_weights=None):
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
            if self.edge_model:
                return edge_features
            else:
                return out


class BaseEdgeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals, **model_kwargs):
        super(BaseEdgeModel,self).__init__()
        self.residuals = residuals
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features, hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, n_targets),
        )
        self.edge_model = True

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            try:
                out = out + edge_attr
            except:
                pass # on the first iteration we shouldn't add residuals
        return out


class TimeAwareNodeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals, agg_future: str, agg_past: str,
                 **padding_kwargs):
        super(TimeAwareNodeModel, self).__init__()
        self.residuals = residuals
        self.node_mlp_future = nn.Sequential(
            nn.Linear(n_features + n_edge_features, hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, n_targets),
        )
        self.node_mlp_past = nn.Sequential(
            nn.Linear(n_features + n_edge_features, hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, n_targets),
        )
        self.node_mlp_combine = nn.Sequential(
            nn.Linear(n_targets * 2, hiddens),
            nn.LeakyReLU(),
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
        self.edge_model = False

    def forward(self, x, edge_index, edge_attr, u, batch):
        n1, n2 = edge_index

        # future_mask = n1 < n2
        # future_n1, future_n2 = n1[future_mask], n2[future_mask]
        # future_in = torch.cat([x[future_n2], edge_attr[future_mask]], dim=1)
        future_in = torch.cat([x[n2], edge_attr], dim=1)
        future_out = self.node_mlp_future(future_in)
        # future_out = self.agg_function[self.agg_future](future_out, future_n1, dim=0, dim_size=x.size(0))
        future_out = self.agg_function[self.agg_future](future_out, n1, dim=0, dim_size=x.size(0))

        # past_mask = n1 > n2
        # past_n1, past_n2 = n1[past_mask], n2[past_mask]
        # past_in = torch.cat([x[past_n2], edge_attr[past_mask]], dim=1)
        past_in = torch.cat([x[n1], edge_attr], dim=1)
        past_out = self.node_mlp_past(past_in)
        # past_out = self.agg_function[self.agg_past](past_out, past_n1, dim=0, dim_size=x.size(0))
        past_out = self.agg_function[self.agg_past](past_out, n2, dim=0, dim_size=x.size(0))

        flow = torch.cat((past_out, future_out), dim=1)

        out = self.node_mlp_combine(flow)

        if self.residuals:
            try:
                out = out + x
            except:
                pass # not add residuals on the first it. TODO: find a more elegant way
        return out


class GATv2ConvWithEdgeUpdate(GATv2Conv):

    def __init__(
        self,
            in_channels:int=256, out_channels:int=256,
            dropout:float=.3, agg_base:str='mean',
            n_edge_features:int=6,
             heads:int=6,
        **padding_kwargs,
    ):
        super(GATv2ConvWithEdgeUpdate,self).__init__(node_dim=0,
                                                     in_channels=in_channels, out_channels=out_channels,
                                                      heads=heads, dropout=dropout, add_self_loops=False,
                                                      edge_dim=n_edge_features, fill_value=agg_base)

        self.concat = False
        self.negative_slope = 0.4


        self.lin_l = Linear(self.in_channels, heads * self.out_channels,
                            weight_initializer='glorot')

        self.lin_r = Linear(self.in_channels, heads * self.out_channels,
                            weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, self.out_channels))

        self.bias = Parameter(torch.Tensor(self.out_channels))

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        #£ type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        #£ type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        #£ type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        #£ type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None


        out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            if self.edge_model:
                edge_attr = self.__edge_attr__
                self.__edge_attr__ = None
                return edge_attr
            else:
                return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            x += edge_attr  # ----> edges are updeted here <-----
            self.__edge_attr__ = edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


# todo: integrate aggregation in other models
def build_custom_mp(n_target_nodes, n_target_edges, n_features, n_edge_features, layer_size, residuals, model_dict:dict,
                    future_aggregation:str='sum', past_aggregation='mean',base_aggregation='sum',device="cuda",
                    heads:int=3, dropout:float=.3 ):
    edge_model = model_dict['edge'](n_features=n_features, n_edge_features=n_edge_features, hiddens=layer_size,
                                    n_targets=n_target_edges, residuals=residuals)
    node_model = model_dict['node'](n_features=n_features, n_edge_features=n_target_edges, hiddens=layer_size,
                                    n_targets=n_target_nodes, residuals=residuals,
                                    agg_future=future_aggregation, agg_past=past_aggregation, agg_base=base_aggregation,
                                    in_channels=n_features, out_channels=n_target_nodes, heads=heads, dropout=dropout)
    return torch_geometric.nn.MetaLayer(
        edge_model=edge_model,
        node_model=node_model
    ).to(device=device)



class Net(torch.nn.Module):

    def __init__(self,
                 node_features_dim:int,
                 model_dict: dict,
                 layer_size=256,
                 n_target_nodes=256,
                 n_target_edges=256,
                 heads:int=3,
                 dropout:float=.3,
                 steps=2,
                 edge_features_dim=EDGE_FEATURES_DIM,
                 residuals: bool = True,
                 past_aggregation:str="mean",
                 future_aggregation:str='sum',
                 base_aggregation:str='mean',
                 dtype=torch.float32,
                 mps_fallback=False,
                 device='cuda:0',
                 is_edge_model: bool = True,
                 used_backbone:str = 'resnet50',
                 **kwargs):
        super(Net, self).__init__()
        self.layer_size = layer_size
        self.node_features_dim = node_features_dim
        self.model_dict = model_dict
        self.is_edge_model = is_edge_model
        self.used_backbone = used_backbone
        self.past_aggregation = past_aggregation
        self.future_aggregation = future_aggregation
        self.base_aggregation = base_aggregation

        self.conv_in = build_custom_mp(n_target_nodes=n_target_nodes, n_target_edges=n_target_edges, n_features=self.node_features_dim, n_edge_features=edge_features_dim,
                                       layer_size=layer_size, residuals=residuals, device=device, model_dict=model_dict, future_aggregation=future_aggregation,
                                       past_aggregation=past_aggregation, base_aggregation=base_aggregation, heads=heads, dropout=dropout)

        kwargs['edge_dim'] = layer_size
        kwargs['in_edge_channels'] = layer_size * heads

        self.number_of_message_passing_layers = steps
        if self.model_dict['node_name'] != 'timeaware':
            in_features = n_target_nodes * heads
        else:
            in_features = n_target_nodes
        self.conv = []
        for i in range(steps - 1):
            # self.conv.append(self.layer_aliases[layer_tipe](in_channels=-1, out_channels=layer_size, **kwargs))
            self.conv.append(
                build_custom_mp(n_target_nodes=n_target_nodes, n_target_edges=n_target_edges, n_features=in_features, n_edge_features=n_target_edges,
                                layer_size=layer_size, residuals=residuals, device=device, model_dict=model_dict, heads=heads, dropout=dropout)
            )

        # self.predictor = EdgePredictor(layer_size, layer_size)
        if is_edge_model:
            self.predictor = EdgePredictorFromEdges(in_channels=n_target_edges, out_channels=layer_size)
        else:
            self.predictor = EdgePredictorFromNodes(in_channels=n_target_nodes, out_channels=layer_size)
        self.dtype = dtype
        self.device = device  # get the device the model is currently on
        self.mps_fallback = mps_fallback
        self.to(dtype=dtype)

        # Fallback to CPU if device is MPS
        if self.mps_fallback:
            print('[INFO] Falling back to CPU for conv layers.')

    def forward(self, data) -> tuple:

        # Step 1 - Extract features from the image
        x, edge_index, edge_attr = data.detections, data.edge_index, data.edge_attr
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

        x, edge_attr, _ = self.conv_in(x=x, edge_index=edge_index.to(torch.int64), edge_attr=edge_attr)

        for i in range(self.number_of_message_passing_layers - 1):
            x, edge_attr, _ = self.conv[i](x=x, edge_index=edge_index.to(torch.int64), edge_attr=edge_attr)

        # Back on MPS after the convolutions
        if self.mps_fallback:
            x = x.to(torch.device('mps'))
            edge_attr = edge_attr.to(torch.device('mps'))

        # Use edge_index to find the corresponding node features in x
        x_i, x_j = x[edge_index[0].to(torch.int64)], x[edge_index[1].to(torch.int64)]
        if self.is_edge_model:
            return self.predictor(edge_attr)
        else:
            return self.predictor(x_i, x_j)

    def __str__(self):
        model_type = "edge-predictor" if self.is_edge_model else "node-predictor"
        node_model = "node-model-" + self.model_dict['node_name']
        edge_model = "edge-model-" + self.model_dict['edge_name']
        layer_size = 'layer-size-' + str(self.layer_size)
        backbone = "backbone-" + self.used_backbone
        if ('timeaware' in self.model_dict['node_name']) or ('timeaware' in self.model_dict['edge_name']):
            aggregation = 'past-'+self.past_aggregation + "-future-"+ self.future_aggregation
        else:
            aggregation = 'aggregation-' + self.base_aggregation
        name = '_'.join([model_type,node_model,edge_model,layer_size,backbone,aggregation])
        return name

IMPLEMENTED_MODELS = {
    'timeaware':{
        'node': TimeAwareNodeModel,
        'edge': BaseEdgeModel,
        'node_name':'timeaware',
        'edge_name':'base'
    },
'transformer':{
        'node': TransformerConvWithEdgeUpdate,
        'edge': BaseEdgeModel,
        'node_name':'transformer',
        'edge_name':'base'
    },
'attention':{
        'node': GATv2ConvWithEdgeUpdate,
        'edge': BaseEdgeModel,
        'node_name':'attention',
        'edge_name':'base'
    }
}