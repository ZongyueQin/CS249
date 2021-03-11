import os
import argparse
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

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
args.n_perturbations = 1000
args.max_no_increase_epoch_num = 50
#args.node_dim = 9
#args.edge_dim = 3
#args.bsz      = 128

print('w_robust = %d'%args.w_robust)
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

data = pre_process_no_batch(pyg_data).to(device)

#dataset = PygGraphPropPredDataset(name = args.dataset, root='dataset/')
#evaluator = Evaluator(name=args.dataset)
#print(evaluator.expected_input_format)
#print(evaluator.expected_output_format)
def trades_on_edge(data, model, train = True):
    
    step_size = 1e-2
    original_adv_atts = []
    for _ in model.gcs:
        atts = Variable(data.edge_attr.repeat(args.n_heads, 1).t(), requires_grad=False).to(args.device)
        if not train:
            atts = torch.round(atts).float()
        original_adv_atts.append(atts)
    if not train:
        return original_adv_atts, None
    perturb_adv_atts  = []
    for _ in model.gcs:
        rand_att = 1e-3 * torch.randn(original_adv_atts[0].shape).to(args.device)
        perturb_adv_atts += [Variable(rand_att.data, requires_grad=True)]
    model.eval()
    for i in range(10):
        ori_out = model(data.x, data.edge_index, data.batch, original_adv_atts)
        adv_out = model(data.x, data.edge_index, data.batch, [torch.clamp(ori_att + adv_att, min=0.0001)\
                                for ori_att, adv_att in zip(original_adv_atts, perturb_adv_atts)])
        loss = criterion_kl(turn_prob(ori_out).log(), turn_prob(adv_out))
        grad = torch.autograd.grad(loss, perturb_adv_atts)
        for g, adv_att in zip(grad, perturb_adv_atts):
            adv_att.data = adv_att.detach() + step_size * torch.sign(g.detach())
    return original_adv_atts, [torch.clamp(ori_att + adv_att, min=0.0001).detach() for ori_att, adv_att \
                                    in zip(original_adv_atts, perturb_adv_atts)]

""" for PGDAttack, but currently there are problems """
"""
class Wrp_GNN(nn.Module):
    def __init__(self, model):
        super(Wrp_GNN, self).__init__()
        self.model = model
        self.nclass = model.n_out
        self.nfeat = model.n_ins[0]
        self.hidden_sizes = model.n_ins[1:]
        self.output = None
         
    def forward(self, features, adj):
        edge_index, _ = from_scipy_sparse_matrix(scipy.sparse.csr_matrix(adj))
        new_edge_index, init_att = randomly_perturb(node_size = features.shape[0], edge_index = edge_index)
        original_adv_atts = []
        for _ in self.model.gcs:
            original_adv_atts += [Variable(init_att.repeat(args.n_heads, 1).t(), requires_grad=False).to(device)]
        d = Data(x = features, edge_index = new_edge_index)
        d = d.to(device)
        return self.model(d.x, d.edge_index, None, original_adv_atts)
"""
model = GNN2(num_feats, args.n_hids, 
            num_classes, args.n_heads, 
            args.n_layer, args.dropout,
            node_level = True).to(device)
criterion = torch.nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss(reduction='sum')

optimizer = get_optimizer(model, weight_decay=args.weight_decay, learning_rate=1e-2)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start = 0.05,\
        steps_per_epoch=args.step_per_epoch, epochs = args.num_epochs, anneal_strategy = 'linear')

highest_val_acc = 0.
no_increase_epoch_num = 0
for epoch in range(args.num_epochs):
    train_loss = []
    train_robust = []
    for i in range(args.step_per_epoch):
        ori_adv_atts, adv_adv_atts = trades_on_edge(data, model)
        model.train()
        ori_out = model(data.x, data.edge_index, None, ori_adv_atts)
        adv_out = model(data.x, data.edge_index, None, adv_adv_atts)
        ori_out_train = ori_out[data.train_mask]
        y_train = data.y[data.train_mask]
        loss = criterion(ori_out_train, y_train)
        loss_robust = criterion_kl(turn_prob(ori_out).log(), turn_prob(adv_out))/data.num_nodes
        (loss + args.w_robust * loss_robust).backward()

        train_robust.append(loss_robust.item())
        train_loss.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        ori_adv_atts, _ = trades_on_edge(data, model, train = False)
        out = model(data.x, data.edge_index, None, ori_adv_atts)
        out_valid = out[data.val_mask]
        y_valid = data.y[data.val_mask]
        valid_loss = criterion(out_valid, y_valid)
        predict = torch.argmax(out_valid, dim = 1)
        correct = (predict.view(-1).long()==y_valid.long()).sum()
        total = data.val_mask.sum()
        valid_acc = correct.float() / total.float()

        if valid_acc > highest_val_acc:
            highest_val_acc = valid_acc
            no_increase_epoch_num = 0
        else:
            no_increase_epoch_num = no_increase_epoch_num + 1
        if no_increase_epoch_num >= args.max_no_increase_epoch_num:
            print('early stop')
            break

        ori_adv_atts, _ = trades_on_edge(data, model, train = False)
        out = model(data.x, data.edge_index, None, ori_adv_atts)
        out_train = out[data.train_mask]
        y_train = data.y[data.train_mask]
        predict = torch.argmax(out_train, dim = 1)
        correct = (predict.view(-1).long()==y_train.long()).sum()
        total = data.train_mask.sum()
        train_acc = correct.float() / total.float()

    print('Epoch %d: LR: %.5f, Train loss: %.3f Train Accuracy: %.3f Valid loss: %.3f  Valid Accuracy: %.3f' \
          % (epoch, optimizer.param_groups[0]['lr'], np.average(train_loss), train_acc, valid_loss, valid_acc))

model.eval()
with torch.no_grad():
    ori_adv_atts, _ = trades_on_edge(data, model, train = False)
    out = model(data.x, data.edge_index, None, ori_adv_atts)
    out_test = out[data.test_mask]
    y_test = data.y[data.test_mask]
    predict = torch.argmax(out_test, dim = 1)
    correct = (predict.view(-1).long()==y_test.long()).sum()
    total = data.test_mask.sum()
    acc = correct.float() / total.float()
    print('Test Accuracy: %.3f' % (acc))

    """ Robust Accuracy (Random) """
    perturbed_adj = apply_Random(dpr_data.adj, n_perturbations = args.n_perturbations) 
    wrp_pyg_data.update_edge_index(perturbed_adj)
    adv_data = pre_process_no_batch(wrp_pyg_data.data).to(device)
    ori_adv_atts, _ = trades_on_edge(adv_data, model, train = False)
    out = model(adv_data.x, adv_data.edge_index, None, ori_adv_atts)
    out_test = out[adv_data.test_mask]
    y_test = adv_data.y[adv_data.test_mask]
    predict = torch.argmax(out_test, dim = 1)
    correct = (predict.view(-1).long()==y_test.long()).sum()
    total = adv_data.test_mask.sum()
    acc = correct.float() / total.float()
    print('Test Robust Accuracy (Random): %.3f' % (acc))

    """ Robust Accuracy (DICE) """
    perturbed_adj = apply_DICE(dpr_data.adj, dpr_data.labels, n_perturbations = args.n_perturbations) 
    wrp_pyg_data.update_edge_index(perturbed_adj)
    adv_data = pre_process_no_batch(wrp_pyg_data.data).to(device)
    ori_adv_atts, _ = trades_on_edge(adv_data, model, train = False)
    out = model(adv_data.x, adv_data.edge_index, None, ori_adv_atts)
    out_test = out[adv_data.test_mask]
    y_test = adv_data.y[adv_data.test_mask]
    predict = torch.argmax(out_test, dim = 1)
    correct = (predict.view(-1).long()==y_test.long()).sum()
    total = adv_data.test_mask.sum()
    acc = correct.float() / total.float()
    print('Test Robust Accuracy (DICE): %.3f' % (acc))

    """ Test Robust Accuracy (PGD) """
    
    """
    wrp_model = Wrp_GNN(model)
    perturbed_adj = apply_PGDAttack(wrp_model, dpr_data.features, dpr_data.adj,
                                    dpr_data.labels, dpr_data.idx_train, 
                                    n_perturbations = 30) 
    pyg_data.update_edge_index(perturbed_adj)
    adv_data = pre_process_no_batch(pyg_data.data).to(device)
    ori_adv_atts, _ = trades_on_edge(adv_data, model, train = False)
    out = model(adv_data.x, adv_data.edge_index, None, ori_adv_atts)
    out_test = out[adv_data.test_mask]
    y_test = adv_data.y[adv_data.test_mask]
    predict = torch.argmax(out_test, dim = 1)
    correct = (predict.view(-1).long()==y_test.long()).sum()
    total = adv_data.test_mask.sum()
    acc = correct.float() / total.float()
    print('Test Robust Accuracy (PGDAttack): %.3f' % (acc))
    """
    """
    for rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
        perturbed_data = PrePtbDataset(root = 'dataset/',
                                           name = 'cora',
                                           attack_method = 'meta',
                                           ptb_rate = rate)
        ptb_adj = perturbed_data.adj
        pyg_data.update_edge_index(ptb_adj)
        adv_data = pre_process_no_batch(pyg_data.data).to(device)
        ori_adv_atts, _ = trades_on_edge(adv_data, model, train = False)
        out = model(adv_data.x, adv_data.edge_index, None, ori_adv_atts)
        out_test = out[adv_data.test_mask]
        y_test = adv_data.y[adv_data.test_mask]
        predict = torch.argmax(out_test, dim = 1)
        correct = (predict.view(-1).long()==y_test.long()).sum()
        total = adv_data.test_mask.sum()
        acc = correct.float() / total.float()
        print('Test Robust Accuracy (Preperturbed Method: Meta rate: %.2f): %.3f' % (rate, acc))
   
    for rate in [1.0, 2.0, 3.0, 4.0, 5.0]:
        perturbed_data = PrePtbDataset(root = 'dataset/',
                                           name = 'cora',
                                           attack_method = 'nettack',
                                           ptb_rate = rate)
        ptb_adj = perturbed_data.adj
        pyg_data.update_edge_index(ptb_adj)
        adv_data = pre_process_no_batch(pyg_data.data).to(device)
        ori_adv_atts, _ = trades_on_edge(adv_data, model, train = False)
        out = model(adv_data.x, adv_data.edge_index, None, ori_adv_atts)
        out_test = out[adv_data.test_mask]
        y_test = adv_data.y[adv_data.test_mask]
        predict = torch.argmax(out_test, dim = 1)
        correct = (predict.view(-1).long()==y_test.long()).sum()
        total = adv_data.test_mask.sum()
        acc = correct.float() / total.float()
        print('Test Robust Accuracy (Preperturbed Method: Nettack rate: %.2f): %.3f' % (rate, acc))
    """ 



#torch.save(model, 'GCN_Cora.pkl')
torch.save(model.state_dict(), 'GCN_2_Cora_parameters.pkl')
