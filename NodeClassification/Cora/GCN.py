import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import math
from torch_geometric.nn import global_mean_pool



class GNN(nn.Module):
    def __init__(self, n_hid, n_out, n_heads, n_layers, dropout = 0.2, node_level = False):
        super(GNN, self).__init__()
        self.node_encoder = AtomEncoder(emb_dim=n_hid)
        self.n_hid     = n_hid
        self.n_out     = n_out
        self.drop      = nn.Dropout(dropout)
        self.gcs       = nn.ModuleList([GCN_Layer(n_hid, n_heads, dropout)\
                                      for _ in range(n_layers)])
        self.out       = nn.Linear(n_hid, n_out)
        self.node_level = node_level

    def forward(self, node_attr, edge_index, batch_idx, adv_atts):
        node_rep = self.node_encoder(node_attr)
        for gc, adv_att in zip(self.gcs, adv_atts):
            node_rep = gc(node_rep, edge_index, adv_att)
        if self.node_level == False:
            return self.out(global_mean_pool(node_rep, batch_idx))  
        else:
            return node_rep

class GNN2(nn.Module):
    def __init__(self, n_feat, n_hids, n_out, n_heads, n_layers, dropout = 0.2, node_level = False):
        super(GNN2, self).__init__()
        #self.node_encoder = AtomEncoder(emb_dim=n_hid)
        #self.node_encoder = nn.Linear(n_feat, n_hid)
        #self.n_hid     = n_hid
        self.n_out     = n_out
        self.n_ins = [n_feat] + n_hids
        self.n_outs = n_hids + [n_out]
        self.drop      = nn.Dropout(dropout)
        self.gcs       = nn.ModuleList([GCN_Layer2(self.n_ins[i], self.n_outs[i], n_heads, dropout)\
                                      for i in range(n_layers)])
        #self.out       = nn.Linear(n_hid, n_out)
        self.node_level = node_level

    def forward(self, node_attr, edge_index, batch_idx, adv_atts):
        #node_rep = self.node_encoder(node_attr)
        node_rep = node_attr
        for gc, adv_att in zip(self.gcs, adv_atts):
            node_rep = gc(node_rep, edge_index, adv_att)
        if self.node_level == False:
            return self.out(global_mean_pool(node_rep, batch_idx))  
        else:
            return node_rep #self.out(node_rep)


class GCN_Layer(MessagePassing):
    def __init__(self, n_hid, n_heads, dropout = 0.2, **kwargs):
        super(GCN_Layer, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.att           = None
        self.a_linear   = nn.Linear(n_hid,   n_hid)
        self.norm       = nn.LayerNorm(n_hid)
        self.drop       = nn.Dropout(dropout)
        
    def forward(self, node_inp, edge_index, adv_att):
        return self.propagate(edge_index, node_inp=node_inp, adv_att = adv_att)

    def message(self, edge_index_i, node_inp_j, adv_att):
        '''
            j: source, i: target; <j, i>
        '''

        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(adv_att.log(), edge_index_i)
        return node_inp_j * self.att.view(-1, 1)


    def update(self, aggr_out, node_inp):
        trans_out = self.norm(self.drop(self.a_linear(F.relu(aggr_out))) + node_inp)
        return trans_out

class GCN_Layer2(MessagePassing):
    def __init__(self, n_in, n_out, n_heads, dropout = 0.2, **kwargs):
        super(GCN_Layer2, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.att           = None
        self.a_linear   = nn.Linear(n_in,   n_out)
        self.norm       = nn.LayerNorm(n_out)
        self.drop       = nn.Dropout(dropout)
        
    def forward(self, node_inp, edge_index, adv_att):
        return self.propagate(edge_index, node_inp=node_inp, adv_att = adv_att)

    def message(self, edge_index_i, node_inp_j, adv_att):
        '''
            j: source, i: target; <j, i>
        '''

        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(adv_att.log(), edge_index_i)
        return node_inp_j * self.att.view(-1, 1)


    def update(self, aggr_out, node_inp):
        trans_out = self.norm(self.drop(self.a_linear(F.gelu(aggr_out)+node_inp)))
        return trans_out
