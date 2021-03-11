import os
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
import torch

def idx2mask(idx, num_nodes):
    mask = np.zeros([num_nodes], dtype=np.bool_)
    mask[idx] = True
    return torch.tensor(mask, dtype=torch.bool)

dataset = PygNodePropPredDataset(name = 'ogbn-arxiv')
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

pyg_data = dataset[0]
pyg_data.train_mask = idx2mask(train_idx, pyg_data.num_nodes)
pyg_data.test_mask = idx2mask(test_idx, pyg_data.num_nodes)
pyg_data.val_mask = idx2mask(valid_idx, pyg_data.num_nodes)

dpr_data = Pyg2Dpr([pyg_data])

