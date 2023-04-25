"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
# gxx
#from dgl.nn import AvgPooling

from dgl.nn.pytorch import AvgPooling

# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

# 注释
# from alignn.models.utils import RBFExpansion
from models.utils import RBFExpansion
from alignn.utils import BaseSettings


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))


        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)

        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        # print(m.shape)
        # center = g.in_degrees() + g.out_degrees()
        # print(center.shape)
        # center = torch.tensor(center, dtype=torch.float32)
        # center = center.cuda()
        # center = F.softmax(center, dim=0)
        # center = center.unsqueeze(1).expand(m.shape)
        # m = m * center


        g.edata["sigma"] = torch.sigmoid(m)         # 聚合了边、节点和上层的三维特征
        g.ndata["Bh"] = self.dst_update(node_feats)       #初始的特征
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")



        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.relu6(self.bn_nodes(x))
        y = F.relu6(self.bn_edges(m))
        # x = F.silu(self.bn_nodes(x))
        # y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class ALIGNNConvNoAngle(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        return x, y


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            # nn.SiLU(),
            nn.ReLU6(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


import copy
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, dim_model):
        super(MultiHeadAttention, self).__init__()
        self.d_k = dim_model//h
        self.h = h
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)

    def get(self, x, fields='qkv'):
        batch_size = x.shape[0]
        ret = {}
        if 'q' in fields:
            ret['q'] = self.linears[0](x).view(batch_size, self.h, self.d_k)
        if 'k' in fields:
            ret['k'] = self.linears[1](x).view(batch_size, self.h, self.d_k)
        if 'v' in fields:
            ret['v'] = self.linears[2](x).view(batch_size, self.h, self.d_k)
        return ret

    def get_o(self, x):
        batch_size = x.shape[0]
        # print(x.shape)
        return self.linears[3](x.view(batch_size, -1))


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x*self.relu6(x+3)/6

class PositionwiseFeedForward(nn.Module):
    '''
    This module implements feed-forward network(after the Multi-Head Network) equation:
    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    '''
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Transformer(nn.Module):
    def __init__(self, in_dim = 256, hidden_dim = 256):
        super(Transformer, self).__init__()
        self.h = 4
        self.d_k = 64
        self.linearQ = nn.Linear(in_dim, hidden_dim)
        self.linearK = nn.Linear(in_dim, hidden_dim)
        self.linearV = nn.Linear(in_dim, hidden_dim)
        self.attention = MultiHeadAttention(h=4, dim_model=hidden_dim)
        self.bn_edges = nn.BatchNorm1d(hidden_dim)
        self.dropout  = nn.Dropout(0.1)

        self.SEblock = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, hidden_dim),
            HardSwish(inplace=True),
        )
        self.ffn = PositionwiseFeedForward(hidden_dim, self.d_k)


        #self.coords = coords
        #self.centrality = centrality
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.linearQ.weight)
        nn.init.xavier_uniform(self.linearK.weight)
        nn.init.xavier_uniform(self.linearV.weight)
        # nn.init.xavier_uniform(self.linear.weight)

    def src_dot_dst(self, src_field, dst_field, out_field):
        def func(edges):
            return {out_field:(edges.src[src_field]*edges.dst[dst_field]).sum(-1, keepdim=True)}
        return func

    def scaled_exp(self, field, scale_constant):
        def func(edges):
            return {field: torch.exp((edges.data[field]/scale_constant).clamp(-5, 5))}
        return func

    def forward(self, g, v):
        mV = v

        center = g.in_degrees() + g.out_degrees()
        #print(center.shape)
        center = torch.as_tensor(center, dtype=torch.float32)
        center = center.cuda()
        center = F.softmax(center, dim=0)
        center = center.unsqueeze(1).expand(mV.shape)
        mV = mV * center


        numberSize = mV.shape[0]
        g.ndata['q'] = self.linearQ(mV).view(numberSize, self.h, self.d_k)
        g.ndata['k'] = self.linearK(mV).view(numberSize, self.h, self.d_k)
        g.ndata['v'] = self.linearV(mV).view(numberSize, self.h, self.d_k)
        # print(g.ndata['q'].shape)
        g.apply_edges(self.src_dot_dst('k', 'q', 'score'))

        # testMid = g.edata['score']
        # testMid = F.normalize(testMid,  dim=1)     # L2正则化
        # g.edata['score'] = testMid

        g.apply_edges(self.scaled_exp('score', np.sqrt(self.d_k)))

        g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

        # x = startFeat
        wv = g.ndata['wv']
        z = g.ndata['z']

        o = self.attention.get_o(wv/z)
        o = self.dropout(o)

        x = F.relu6(self.bn_edges(o))

        # x = self.SEblock(x)
        # # x = seX * x
        seX = self.SEblock(x)
        x = seX * x


        # x = self.ffn(x)

        #y = F.relu6(self.bn_edges(m))
        x = x + v
        #y = y + e
        return x





class Tran_GCN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )
        # self.coords_embedding = nn.Linear(3, 32)
        # self.mid_embedding = MLPLayer(288, 256)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )

        # self.alignn_layers = nn.ModuleList(
        #     [
        #         ALIGNNConvNoAngle(
        #             config.hidden_features,
        #             config.hidden_features,
        #         )
        #         for idx in range(config.alignn_layers)
        #     ]
        # )

        # self.gcn_layers = nn.ModuleList(
        #     [
        #         EdgeGatedGraphConv(
        #             config.hidden_features, config.hidden_features
        #         )
        #         for idx in range(config.gcn_layers)
        #     ]
        # )


        # transformer
        self.tran1 = Transformer()
        self.tran2 = Transformer()
        self.tran3 = Transformer()

        # self.tran4 = Transformer()
        # self.tran5 = Transformer()

        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            # print(lg.edata["h"].shape)

            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()



        # initial node features: atom feature network...
        # 原子特征
        x = g.ndata.pop("atom_features")
        # x = torch.cat((x, coords), dim=1)
        x = self.atom_embedding(x)


        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        x = self.tran1(g, x)
        x = self.tran2(g, x)
        x = self.tran3(g, x)

        # x = self.tran4(g, x)
        # x = self.tran5(g, x)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)


        # # 不包括键角的特征
        # for alignn_layer in self.alignn_layers:
        #     x, y = alignn_layer(g, lg, x, y)

        # gated GCN updates: update node, edge features

        # for gcn_layer in self.gcn_layers:
        #     x, y = gcn_layer(g, x, y)


        # 考虑坐标特征
        # 坐标特征信息
        # coords = g.ndata.pop("coords")
        # coordsNorm = coords.reshape(-1)
        # mean = coordsNorm.mean(dim=0)
        # std = coordsNorm.std(dim=0)
        # coords = (coords-mean)/std
        # coords = self.coords_embedding(coords)

        # x = torch.cat((x, coords), dim=1)
        # x = self.mid_embedding(x)




        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)
