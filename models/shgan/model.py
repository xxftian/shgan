import torch
import torch.nn as nn

import torch.optim as optim
import dgl

from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
from dgl.data.utils import save_graphs, load_graphs, _get_dgl_url
from dgl.convert import heterograph
from dgl.data import DGLBuiltinDataset
from dgl import backend as F
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.utils import early_stopper

import numpy as np
import pandas as pd
import random
import yaml
import json

random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed_all(2023)


def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters. \n')


class EdgeEmbedding(nn.Module):
    def __init__(self, args):
        super(EdgeEmbedding, self).__init__()
        self.feature_embeddings = nn.ModuleDict()
        self.feature_idx = {int(k): v['name'] for k, v in args['mdatas'].items()}
        self.embed_dim = args['embed_dim']
        self.hidden_dim = args['hidden_dim']
        self.numers_len = sum([1 for k, v in args['mdatas'].items() if v['dtype'] == 'numerical'])
        self.linear1 = nn.Linear(self.numers_len, self.embed_dim, bias=False)
        self.linear2 = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        for feat_id, mdata in args['mdatas'].items():
            if mdata['dtype'] == 'categorical':
                self.feature_embeddings[mdata['name']] = nn.Embedding(mdata['max_size'] + 1, self.embed_dim,
                                                                      padding_idx=0).to(args['device'])
            else:
                pass
        self.to(args['device'])

    def forward(self, edge_feats):
        num_inputs, obj_inputs = [], []
        for idx in range(edge_feats.shape[1]):
            if self.feature_idx[idx] in self.feature_embeddings:
                edge_feat = edge_feats[:, idx].type(torch.long)
                mask = (edge_feat == 0).unsqueeze(-1)
                embeded = self.feature_embeddings[self.feature_idx[idx]](edge_feat)
                embeded[mask.expand_as(embeded)] = torch.zeros_like(embeded[mask.expand_as(embeded)])
                obj_inputs.append(embeded)
            else:
                num_inputs.append(edge_feats[:, idx])

        obj_inputs = torch.stack(obj_inputs, dim=-1).sum(-1)
        num_inputs = torch.stack(num_inputs, dim=-1)
        num_inputs = self.linear1(num_inputs)

        return self.linear2(obj_inputs + num_inputs)  # batch x embed_dim


class GatedResidual(nn.Module):
    def __init__(self, dim, only_gate=False, rate=0.1):
        super(GatedResidual, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(rate)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.ReLU()
        self.only_gate = only_gate

    def forward(self, x1, x2):
        gate_input = torch.cat((x1, x2, x1 - x2), dim=-1)
        gate = self.proj(gate_input)
        if self.only_gate:
            return x1 * gate + x2 * (1 - gate)
        return self.norm(self.drop(x1 * gate + x2 * (1 - gate)))


class GraphMultiAttentionV2(nn.Module):
    def __init__(self, feat_size, num_heads, attn_dropout, device, use_efeats=False):
        super(GraphMultiAttentionV2, self).__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.use_efeats = use_efeats
        self.device = device
        assert self.head_dim * num_heads == feat_size

        self.q_proj = nn.Linear(feat_size, feat_size, bias=False)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=False)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=False)
        self.node_proj = nn.Linear(feat_size, feat_size, bias=False)

        self.edge_input = nn.Linear(feat_size, self.num_heads, bias=False)
        self.gate = nn.Linear(feat_size, self.num_heads, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)

        nn.init.xavier_uniform_(self.node_proj.weight)
        if self.node_proj.bias is not None:
            nn.init.constant_(self.node_proj.bias, 0.0)

    def forward(self, graph, feat, return_attn=False):
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        q_src = self.q_proj(h_src).view(-1, self.num_heads, self.head_dim)
        k_dst = self.k_proj(h_dst).view(-1, self.num_heads, self.head_dim)
        v_src = self.v_proj(h_src).view(-1, self.num_heads, self.head_dim)

        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        if self.use_efeats:
            e_bias = self.edge_input(graph.edata['feat'])
            gates = torch.sigmoid(self.gate(graph.edata['feat']))
        else:
            e_bias = torch.zeros_like(graph.edata['a'], device=self.device, requires_grad=False)
            gates = torch.ones_like(graph.edata['a'], device=self.device, requires_grad=False)

        graph.edata['a'] = graph.edata['a'].clamp(-5, 5) + e_bias.unsqueeze(-1)
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'] / self.scaling) * gates.unsqueeze(-1)
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'), fn.sum('attn', 'agg_u'))  # 都是element-wise
        rst = graph.dstdata['agg_u'].reshape(-1, self.num_heads * self.head_dim)
        rst = self.node_proj(rst)

        if return_attn:
            return rst, graph.edata['sa']
        else:
            return rst


class GraphTransformerLayer(nn.Module):
    def __init__(self,
                 feat_size,
                 num_heads,
                 bias=False,
                 allow_zero_in_degree=False,
                 attn_dropout=0.2,
                 ffn_dropout=0.1,
                 device='cpu',
                 use_efeats=False):

        super(GraphTransformerLayer, self).__init__()
        self._allow_zero_in_degree = allow_zero_in_degree
        self.hidden_size = feat_size

        self.attn_layer = GraphMultiAttentionV2(feat_size, num_heads, attn_dropout=attn_dropout,
                                                device=device, use_efeats=use_efeats)
        self.residual_layer = GatedResidual(feat_size, only_gate=False)
        self.skip_layer = nn.Linear(feat_size, feat_size, bias=bias)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Linear(feat_size, self.hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(self.hidden_size, feat_size, bias=bias),
            nn.Dropout(p=ffn_dropout)
        )

    def forward(self, graph, feat, get_attention=False):
        graph = graph.local_var()
        skip_feat = self.skip_layer(feat[:graph.number_of_dst_nodes()])

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        if get_attention:
            rst, attns = self.attn_layer(graph, feat, return_attn=True)
        else:
            rst = self.attn_layer(graph, feat)

        rst = self.residual_layer(rst, skip_feat)
        rst = rst + self.ffn(rst)

        return rst


class Tabular1DCNN(nn.Module):
    def __init__(self, input_dim, embed_dim, K=4, dropout=0.1, bias=False):
        super(Tabular1DCNN, self).__init__()
        self.K = K
        self.hid_dim = input_dim * embed_dim * 2
        self.cha_input = input_dim
        self.cha_output = embed_dim
        self.sign_size1 = 2 * embed_dim
        self.sign_size2 = embed_dim

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(input_dim, self.hid_dim, bias=bias)

        self.bn_cv1 = nn.BatchNorm1d(self.cha_input)
        self.conv1 = nn.Conv1d(
            in_channels=self.cha_input,
            out_channels=self.cha_input * self.K,
            kernel_size=5,
            padding=2,
            groups=self.cha_input,  # 这里做的是特殊的深度卷积 inchannels | outchannels | groups
            bias=False
        )
        self.ave_pool1 = nn.AdaptiveAvgPool1d(self.sign_size2)

        self.bn_cv2 = nn.BatchNorm1d(self.cha_input * self.K)
        self.dropout2 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=self.cha_input * self.K,
            out_channels=self.cha_input * self.K,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn_cv3 = nn.BatchNorm1d(self.cha_input * self.K)
        self.conv3 = nn.Conv1d(
            in_channels=self.cha_input * self.K,
            out_channels=self.cha_output,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.final_mlp = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x):
        x = self.dropout1(self.bn1(x))
        x = nn.functional.celu(self.dense1(x))
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)  # batch x input_dim x (2 * embed_dim)

        x = self.bn_cv1(x)
        x = nn.functional.relu(self.conv1(x))  # 这里动的是channels batch x (input_dim * K) x (2 * embed_dim)
        x = self.ave_pool1(x)  # batch x (input_dim * K) x embed_dim

        x_input = x
        x = self.dropout2(self.bn_cv2(x))
        x = nn.functional.relu(self.conv2(x))  # batch x (input_dim * K) x embed_dim
        x = x + x_input

        x = self.bn_cv3(x)
        x = nn.functional.relu(self.conv3(x))  # batch x input_dim x embed_dim

        x = self.final_mlp(x).squeeze(-1)

        return x


class Model(nn.Module):
    def __init__(self, args, dropout=(0.1, 0.1, 0.1)):
        super(Model, self).__init__()

        self.in_feats = args['num_feats']
        self.obj_tuples = args['obj_tuples']
        self.n_classes = args['n_classes']
        self.num_heads = args['num_heads']
        self.embed_dim = args['embed_dim']
        self.hidden_dim = args['hidden_dim']
        self.n_layers = args['n_layers']
        self.device = args['device']
        self.post_proc = args['post_proc']
        self.use_cat = args['use_cat'][args['dataset']]
        self.use_efeats = args['use_efeats'][args['dataset']]
        self.bias = True

        self.obj_table = nn.ModuleDict(
            {col: nn.Embedding(max_value + 1, self.embed_dim).to(self.device) for col, max_value in self.obj_tuples})
        self.obj_list = [col for col, _ in self.obj_tuples]
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim, bias=False) for _ in range(len(self.obj_list))])

        self.edge_emb = EdgeEmbedding(args)

        self.n2v_mlp = Tabular1DCNN(self.in_feats, self.embed_dim)
        self.dropout_emb = nn.Dropout(dropout[0])
        self.dropout_hid = nn.Dropout(dropout[1])
        self.dropout_out = nn.Dropout(dropout[2])

        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            self.n_classes + 1, self.hidden_dim, padding_idx=self.n_classes))
        self.layers.append(nn.Linear(self.embed_dim + self.embed_dim, self.hidden_dim, bias=False))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim),
                                         nn.PReLU(),
                                         nn.Dropout(dropout[2]),
                                         nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                                         ))

        for _ in range(self.n_layers):
            self.layers.append(GraphTransformerLayer(
                feat_size=self.hidden_dim,
                num_heads=self.num_heads,  # 这里先固定写为4
                bias=False,
                device=self.device,
                use_efeats=self.use_efeats
            ))

        if self.post_proc:
            self.layers.append(
                nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                              nn.BatchNorm1d(self.hidden_dim),
                              nn.PReLU(),
                              nn.Dropout(dropout[2]),
                              nn.Linear(self.hidden_dim, self.n_classes, bias=False)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim, self.n_classes, bias=False))

        self.to(self.device)

    def forward_emb(self, cat_col):
        cat_output = 0
        support = {col: self.obj_table[col](cat_col[col]) for col in self.obj_list}

        for i, k in enumerate(support.keys()):
            support[k] = self.dropout_emb(support[k])
            support[k] = self.forward_mlp[i](support[k])
            cat_output = cat_output + support[k]

        return cat_output

    def forward(self, blocks, features, labels, obj_feats):
        """
        features：是原始的数值型输入 float
        obj_feats：用于embedding的累呗性输入 字典型
        """
        trans_h = self.n2v_mlp(features)

        if self.use_cat:
            features = self.forward_emb(obj_feats)

        h = torch.cat((features, trans_h), dim=-1)
        label_embed = self.dropout_hid(self.layers[0](labels))
        # h = self.layers[1](h) + label_embed
        h = self.layers[1](h)
        start_time = time.time()
        for i in range(self.n_layers):
            if self.use_efeats:
                blocks[i].edata['feat'] = self.edge_emb(blocks[i].edata['feat'])

            h = self.dropout_out(self.layers[i + 3](blocks[i], h))

        # print('时间1, {}s'.format(time.time() - start_time))
        logits = self.layers[-1](h)
        h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-6)

        return logits, h


