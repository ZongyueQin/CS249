import torch
import numpy as np
from torch.optim import AdamW, SGD
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
from torch_geometric.data import Data
import math

def turn_prob(inp):
    prob = torch.sigmoid(inp)
    prob = torch.cat([prob, 1-prob], dim=1)
    return prob

def randomly_perturb(node_size, edge_index, ratio = 0.3):
    # add edges
    new_edge = np.random.choice(node_size, 2 * int(edge_index.shape[1] * ratio)).reshape(2, -1)
    init_att = [0.99] * edge_index.shape[1] + [0.01] * len(new_edge[0])
    return torch.cat([edge_index, torch.LongTensor(new_edge)], dim=1), torch.FloatTensor(init_att)

def pre_process(d):
    new_edge_index, init_att = randomly_perturb(node_size = len(d.x), edge_index = d.edge_index)
    return Data(x=d.x, y=d.y, edge_index=new_edge_index, edge_attr=init_att, ori_edge_index = d.edge_index)

def pre_process_no_batch(d):
    new_edge_index, init_att = randomly_perturb(node_size = len(d.x), edge_index = d.edge_index)
    return Data(x=d.x, y=d.y, edge_index=new_edge_index, edge_attr=init_att, ori_edge_index = d.edge_index, batch=None,
                train_mask = d.train_mask, val_mask = d.val_mask, test_mask = d.test_mask, num_nodes = d.num_nodes)

def get_optimizer(model: nn.Module, learning_rate: float = 1e-4, adam_eps: float = 1e-6,
                  weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
#    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    optimizer = SGD(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer
