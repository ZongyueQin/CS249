import os
import argparse
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch_geometric.utils import from_scipy_sparse_matrix
import argparse
import numpy as np
import random

from torch_geometric.data import Data
from torch_geometric.datasets import CoraFull, Planetoid

import deeprobust
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr, PrePtbDataset
#from deeprobust.graph.defense import GCNJaccard, GCNSVD, RGCN, ProGNN
from MyDeepRobustGCN import GCNJaccard, GCNSVD
from MyDeepRobustRGCN import RGCN

from GCN import GNN, GNN2

from utils import *
from attack import apply_Random, apply_DICE, apply_PGDAttack
import scipy.sparse

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.dataset = 'ogbn-arxiv'
#args.n_classes = 10 
args.lr = 1e-2
args.n_hids = [256, 256]
args.n_heads = 1
args.n_layer = len(args.n_hids)+1
args.dropout = 0.
args.num_epochs = 500
args.weight_decay = 0.01
args.w_robust = 0
args.step_per_epoch = 10
args.device = device
args.n_perturbations = [0.01, 0.02, 0.04]
args.max_no_increase_epoch_num = 50
#args.node_dim = 9


print('Loading Data')
print("dataset: {} ".format(args.dataset))

dataset = PygNodePropPredDataset(name = args.dataset)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

pyg_data = dataset[0]
pyg_data.y = pyg_data.y.view(-1)
pyg_data.train_mask = idx2mask(train_idx, pyg_data.num_nodes)
pyg_data.test_mask = idx2mask(test_idx, pyg_data.num_nodes)
pyg_data.val_mask = idx2mask(valid_idx, pyg_data.num_nodes)

dpr_data = Pyg2Dpr([pyg_data])
wrp_pyg_data = Dpr2Pyg(dpr_data)


num_feats = dpr_data.features.shape[1]
num_classes = dpr_data.labels.max().item() + 1
num_edges = pyg_data.edge_index.size(1)

data = pre_process_no_batch(pyg_data).to(device)

adj, features, labels = dpr_data.adj, dpr_data.features.numpy(), dpr_data.labels

print(type(adj))


"""
jaccard = GCNJaccard(nfeat = num_feats, 
                     nhids = args.n_hids,
                     nclass = num_classes,
                     dropout = args.dropout,
                     lr = args.lr,
                     weight_decay = args.weight_decay,
                     device = device).to(device)

jaccard.fit(features, adj, labels, train_idx, valid_idx, threshold=0.03, train_iters = args.num_epochs*args.step_per_epoch)
"""



"""
GCNSVD cannot deal with arxiv dataset, because preprocessed adj is not sparse
svd = GCNSVD(nfeat = num_feats, nhids = args.n_hids,
             nclass = num_classes, dropout = args.dropout,
             lr = args.lr, weight_decay = args.weight_decay,
             device = device).to(device)
svd.fit(features, adj, labels, train_idx, valid_idx, k=20, train_iters = args.num_epochs*args.step_per_epoch)
"""

rgcn = RGCN(nfeat = num_feats, nnodes=features.shape[0],
            nhids=args.n_hids, nclass = num_classes, device = device).to(device)
rgcn.fit(features, adj, labels, train_idx, valid_idx, train_iters = args.num_epochs*args.step_per_epoch)

rgcn_2 = RGCN(nfeat = num_feats, nnodes=features.shape[0],
            nhids=args.n_hids, nclass = num_classes, device = device).to(device)


""" clean accuracy """
#print('GCNJaccard')
#jaccard.test(test_idx)
#print('GCNSVD')
#svd.test(test_idx)
print('RGCN')
rgcn.test(test_idx)

#models = [("GCNJaccard", jaccard)]
models = [("RGCN", rgcn, rgcn_2)]
#models=          [("GCNSVD", svd)]
""" Robust Accuracy (Random) """

y_test = labels[test_idx]
y_test = torch.from_numpy(y_test).long().to(device)
for ratio in args.n_perturbations:
    for name, model, model_2 in models:
        print(name)
        perturbed_adj = apply_Random(dpr_data.adj, n_perturbations = int(ratio*num_edges))
        predict = torch.argmax(model.predict(features = features, adj = perturbed_adj), dim=1)
        predict = predict[test_idx]
        correct = (predict.view(-1).long()==y_test).sum()
        total = pyg_data.test_mask.sum()
        acc = correct.float() / total.float()
        print('Test Robust Accuracy (Random, %.2f): %.3f' % (ratio, acc))
        model_2.fit(features, perturbed_adj, labels, train_idx, valid_idx, train_iters = args.num_epochs*args.step_per_epoch)
        print('posinoning atttack')
        model_2.test(test_idx)

    """ Robust Accuracy (DICE) """
for ratio in args.n_perturbations:
    for name, model, model_2 in models:
        print(name)
        perturbed_adj = apply_DICE(dpr_data.adj, dpr_data.labels, n_perturbations = int(ratio*num_edges))
        predict = torch.argmax(model.predict(features = features, adj = perturbed_adj), dim=1)
        predict = predict[test_idx]
        correct = (predict.view(-1).long()==y_test).sum()
        total = pyg_data.test_mask.sum()
        acc = correct.float() / total.float()
        print('Test Robust Accuracy (DICE, %.2f): %.3f' % (ratio, acc))
        model_2.fit(features, perturbed_adj, labels, train_idx, valid_idx, train_iters = args.num_epochs*args.step_per_epoch)
        print('posinoning atttack')
        model_2.test(test_idx)


