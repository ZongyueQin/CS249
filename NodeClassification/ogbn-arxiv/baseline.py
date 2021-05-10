import os
import argparse
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch_geometric.utils import from_scipy_sparse_matrix
import torch_geometric.transforms as T
import argparse
import numpy as np
import random

from torch_geometric.data import Data
from torch_geometric.datasets import CoraFull, Planetoid

import deeprobust
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr, PrePtbDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax, add_remaining_self_loops, degree, is_undirected, to_undirected
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import math

from utils import *
from attack import apply_Random, apply_DICE, apply_PGDAttack
import scipy.sparse
from preprocess import GCNSVD, GCNJaccard
import GraphAT
from GraphAT import GraphAT_Loss, GraphVAT_Loss, sample_neighbors, EdgeIndex2NeighborLists

# +
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
# -
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.dataset = 'ogbn-arxiv'
#args.n_classes = 10 
args.lr = 1e-2
args.n_hids = 128
args.n_heads = 1
args.n_layer = 3
args.dropout = 0.5
args.num_epochs = 500
args.weight_decay = 0.01
args.w_robust = 0.
args.step_per_epoch = 1
args.device = device
args.n_perturbations = [0.05, 0.1, 0.2, 0.3, 0.4]
args.max_no_increase_epoch_num = 10
#args.node_dim = 9
#args.edge_dim = 3
#args.bsz      = 128
args.use_svd = False
args.use_jaccard = False
args.svd_k = 20
args.jaccard_thres = 0.25
#args.node_dim = 9
#args.edge_dim = 3
#args.bsz      = 128
#arguments for GraphAT
args.num_power_iterations = 1
args.xi = 1e-6
args.epsilon = 1.0
args.epsilon_graph = 0.01
args.num_neighbors = 2
args.gat_w = 0
args.gvat_w = 0

GraphAT.global_args = args

print('w_robust = %f'%args.w_robust)
print('Loading Data')
print("dataset: {} ".format(args.dataset))

#dpr_data = Dataset(root='dataset/', name='cora', seed=15)
#num_feats = dpr_data.features.shape[1]
#num_classes = dpr_data.labels.max().item() + 1
#num_edges = 5429

#pyg_data = Dpr2Pyg(dpr_data)
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


num_feats = pyg_data.num_features#dpr_data.features.shape[1]
num_classes = dataset.num_classes#dpr_data.labels.max().item() + 1
num_edges = pyg_data.edge_index.size(1)
print(num_classes)
#data = pre_process_no_batch(pyg_data).to(device)

adj, features, labels = dpr_data.adj, dpr_data.features.numpy(), dpr_data.labels
# +
#dataset = PygGraphPropPredDataset(name = args.dataset, root='dataset/')
evaluator = Evaluator(name=args.dataset)
#print(evaluator.expected_input_format)
#print(evaluator.expected_output_format)


"""
def pre_process_no_batch(d):
    new_edge_index, init_att = randomly_perturb(node_size = len(d.x), edge_index = d.edge_index)
    return Data(x=d.x, y=d.y, edge_index=new_edge_index, edge_attr=init_att, ori_edge_index = d.edge_index, batch=None,
                train_mask = d.train_mask, val_mask = d.val_mask, test_mask = d.test_mask, num_nodes = d.num_nodes)

def gradient_normalize(g, _type = 'inf'):
    if _type == 'inf':
        return torch.sign(g)
    elif _type == 'l2':
        return args.n_hids * g / g.norm(p=2, keepdim=True)
def trades_on_edge(data, model, train = True, _type = 'l2'):
    step_size = 1e-3
    original_adv_atts = []
    for _ in model.gcs:
        original_adv_atts += [Variable(data.edge_attr.repeat(args.n_heads, 1).t(), requires_grad=False).to(args.device)]
    if not train:
        return original_adv_atts, None
    perturb_adv_atts  = []
    for _ in model.gcs:
        rand_att = 1e-4 * torch.randn(original_adv_atts[0].shape).to(args.device)
        perturb_adv_atts += [Variable(rand_att.data, requires_grad=True)]
    model.eval()
    for i in range(10):
        ori_out = model(data.x, data.edge_index, data.batch, original_adv_atts)
        adv_out = model(data.x, data.edge_index, data.batch, [torch.clamp(ori_att + adv_att, min=0.0001)\
                                for ori_att, adv_att in zip(original_adv_atts, perturb_adv_atts)])
        loss = criterion_kl(turn_prob(ori_out).log(), turn_prob(adv_out))
        grad = torch.autograd.grad(loss, perturb_adv_atts)
        for g, adv_att in zip(grad, perturb_adv_atts):
            n_g = gradient_normalize(g.detach(), _type = _type) 
            adv_att.data = adv_att.detach() + step_size * n_g  
    return original_adv_atts, [torch.clamp(ori_att + adv_att, min=0.0001).detach() for ori_att, adv_att \
                                    in zip(original_adv_atts, perturb_adv_atts)]
"""

if args.use_svd == True:
    assert args.use_jaccard == False
    svd = GCNSVD()
    new_adj = svd.truncatedSVD(dpr_data.adj, k = args.svd_k)
    print(type(new_adj))
    print(np.sum(new_adj!=0))
    pyg_data.update_edge_index(new_adj)
elif args.use_jaccard == True:
    jaccard = GCNJaccard(args.jaccard_thres, binary_feature=False)
    new_adj = jaccard.drop_dissimilar_edges(dpr_data.features.numpy(), dpr_data.adj)
    wrp_pyg_data.update_edge_index(new_adj)
    pyg_data.edge_index = wrp_pyg_data.data.edge_index     

pyg_data.edge_index = to_undirected(pyg_data.edge_index)
model = GNN(num_feats, args.n_hids, \
            num_classes, args.n_layer, \
            args.dropout).to(device)
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.NLLLoss()
criterion_kl = nn.KLDivLoss(reduction='sum')

#optimizer = get_optimizer(model, weight_decay=args.weight_decay, learning_rate=args.lr)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start = 0.05,\
#       steps_per_epoch=args.step_per_epoch, epochs = args.num_epochs + 1, anneal_strategy = 'linear')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def turn_prob(inp):
   # prob = torch.sigmoid(inp)
   # prob = torch.cat([prob, 1-prob], dim=1)
    m, _ = torch.max(inp, dim=1, keepdim=True)
    prob =  F.softmax((inp - m), dim=1)
    #prob = torch.clamp(prob, min=0.001, max=1.)
    return prob


highest_val_acc = 0.
no_increase_epoch_num = 0
train_loss = []
train_robust = [0]
flag = False
neighbor_lists = EdgeIndex2NeighborLists(pyg_data.edge_index,pyg_data.num_nodes, pyg_data.is_directed())
model.reset_parameters()
for epoch in range(args.num_epochs):
    if flag:
        break
    #data = pre_process_no_batch(pyg_data).to(device)
    data = pyg_data.to(device)
    for i in range(args.step_per_epoch):
#        ori_adv_atts, adv_adv_atts = trades_on_edge(data, model, train=True, _type='inf')
        model.train()
#        ori_out = model(data.x, data.edge_index, None, ori_adv_atts)
#        adv_out = model(data.x, data.edge_index, None, adv_adv_atts)
        optimizer.zero_grad()
        ori_out = model(data.x, data.edge_index)
        ori_out_train = ori_out[data.train_mask]
        y_train = data.y[data.train_mask]
        loss = criterion(ori_out_train, y_train)
        if args.gat_w > 0:
            neighbor_ids = sample_neighbors(neighbor_lists)
            loss_gat, _ = GraphAT_Loss(data.x, data.edge_index, model, ori_out,
                                    neighbor_ids, data.train_mask + data.test_mask + data.val_mask)
            loss += args.gat_w * loss_gat
        else:
            loss_gat = 0
        if args.gvat_w > 0:
            loss_gvat = GraphVAT_Loss(data.x, data.edge_index, model, ori_out)
            loss += args.gvat_w * loss_gvat
        else:
            loss_gvat = 0
        """
        loss_robust = criterion_kl(turn_prob(ori_out).log(), turn_prob(adv_out))/data.num_nodes
        if np.isinf(loss_robust.item()):
            print('loss_robust')
            prob = turn_prob(ori_out).cpu().detach().numpy()
            log_prob = np.log(prob)
            is_inf = np.isinf(log_prob)
            print(prob[is_inf])
            print(log_prob[is_inf])
            print(loss_robust.item())
            print(loss.item())
            flag = True
            break
        (loss + args.w_robust * loss_robust).backward()

        train_robust.append(loss_robust.item())
        """
        loss.backward()
        train_loss.append(loss.item())

        optimizer.step()
        #scheduler.step()

    with torch.no_grad():
        #ori_adv_atts, _ = trades_on_edge(data, model, train = False)
        #out = model(data.x, data.edge_index, None, ori_adv_atts)
        model.eval()
        out = model(data.x, data.edge_index)
        out_valid = out[data.val_mask]
        y_valid = data.y[data.val_mask]
        valid_loss = criterion(out_valid, y_valid)
        predict = torch.argmax(out, dim = 1, keepdim=True)
        #correct = (predict.view(-1).long()==y_valid.long()).sum()
        #total = data.val_mask.sum()
        #valid_acc = correct.float() / total.float()

                #ori_adv_atts, _ = trades_on_edge(data, model, train = False)
        #out = model(data.x, data.edge_index, None, ori_adv_atts)
        #out_train = out[data.train_mask]
        #y_train = data.y[data.train_mask]
        #predict = torch.argmax(out_train, dim = 1)
        #correct = (predict.view(-1).long()==y_train.long()).sum()
        #total = data.train_mask.sum()
        #train_acc = correct.float() / total.float()

        #out_test = out[data.test_mask]
        #y_test = data.y[data.test_mask]
        y = data.y.view(-1,1)
        train_acc = evaluator.eval({
        'y_true': y[data.train_mask],
        'y_pred': predict[data.train_mask],
        })['acc']
        valid_acc = evaluator.eval({
        'y_true': y[data.val_mask],
        'y_pred': predict[data.val_mask],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y[data.test_mask],
            'y_pred': predict[data.test_mask],
        })['acc']
        if valid_acc > highest_val_acc:
            highest_val_acc = valid_acc
            no_increase_epoch_num = 0
            torch.save(model.state_dict(), 'GCN_2_Cora_parameters.pkl')
            print('update')


    print('Epoch %d: LR: %.5f, Train loss: %.3f Train Robust: %.3f Train Accuracy: %.3f Valid loss: %.3f  Valid Accuracy: %.3f Test Accuracy: %.3f' \
          % (epoch, optimizer.param_groups[0]['lr'], np.average(train_loss[-100:]), np.average(train_robust[-100:]),\
             train_acc, valid_loss, valid_acc, test_acc))

#criterion_kl(turn_prob(ori_out).log(), turn_prob(adv_out))

#for adv_out in ori_out:
#    if np.isnan(adv_out.cpu().norm().detach().numpy()):
#        print('fail, break')

# +

#ads = []
#for oi, ai in zip(ori_out, adv_out):
#    ads += [criterion_kl(turn_prob(torch.LongTensor([oi])).log(), turn_prob(torch.LongTensor([ai])))]
# -

#adv_out

model.load_state_dict(torch.load('GCN_2_Cora_parameters.pkl'))
model.eval()
with torch.no_grad():
    #ori_adv_atts, _ = trades_on_edge(data, model, train = False)
    #out = model(data.x, data.edge_index, None, ori_adv_atts)
    out = model(data.x, data.edge_index)
    out_test = out[data.test_mask]
    y_test = data.y[data.test_mask]
    predict = torch.argmax(out_test, dim = 1)
    correct = (predict.view(-1).long()==y_test.long()).sum()
    total = data.test_mask.sum()
    acc = correct.float() / total.float()
    print('Test Accuracy: %.3f' % (acc))

    """ Robust Accuracy (Random) """
    for ratio in args.n_perturbations:
        perturbed_adj = apply_Random(dpr_data.adj, n_perturbations = int(ratio*num_edges)) 
        if args.use_svd == True:
            assert args.use_jaccard == False
            perturbed_adj = svd.truncatedSVD(perturbed_adj, k = args.svd_k)
        elif args.use_jaccard == True:
            perturbed_adj = jaccard.drop_dissimilar_edges(dpr_data.features.numpy(), perturbed_adj)

        wrp_pyg_data.update_edge_index(perturbed_adj)
#        adv_data = pre_process_no_batch(wrp_pyg_data.data).to(device)
#        ori_adv_atts, _ = trades_on_edge(adv_data, model, train = False)
#        out = model(adv_data.x, adv_data.edge_index, None, ori_adv_atts)
        adv_data = wrp_pyg_data.data.to(device)
        out = model(adv_data.x, adv_data.edge_index)
        out_test = out[adv_data.test_mask]
        y_test = adv_data.y[adv_data.test_mask]
        predict = torch.argmax(out_test, dim = 1)
        correct = (predict.view(-1).long()==y_test.long()).sum()
        total = adv_data.test_mask.sum()
        acc = correct.float() / total.float()
        print('Test Robust Accuracy (Random %.2f): %.3f' % (ratio, acc))

    """ Robust Accuracy (DICE) """
    for ratio in args.n_perturbations:
        break
        perturbed_adj = apply_DICE(dpr_data.adj, dpr_data.labels, n_perturbations = int(ratio*num_edges)) 
        if args.use_svd == True:
            assert args.use_jaccard == False
            perturbed_adj = svd.truncatedSVD(perturbed_adj, k = args.svd_k)
        elif args.use_jaccard == True:
            perturbed_adj = jaccard.drop_dissimilar_edges(dpr_data.features.numpy(), perturbed_adj)

        wrp_pyg_data.update_edge_index(perturbed_adj)
#        adv_data = pre_process_no_batch(wrp_pyg_data.data).to(device)
#        ori_adv_atts, _ = trades_on_edge(adv_data, model, train = False)
#        out = model(adv_data.x, adv_data.edge_index, None, ori_adv_atts)
        adv_data = wrp_pyg_data.data.to(device)
        out = model(adv_data.x, adv_data.edge_index)
        out_test = out[adv_data.test_mask]
        y_test = adv_data.y[adv_data.test_mask]
        predict = torch.argmax(out_test, dim = 1)
        correct = (predict.view(-1).long()==y_test.long()).sum()
        total = adv_data.test_mask.sum()
        acc = correct.float() / total.float()
        print('Test Robust Accuracy (DICE %.2f): %.3f' % (ratio, acc))


# +
# #torch.save(model, 'GCN_Cora.pkl')
# torch.save(model.state_dict(), 'GCN_2_Cora_parameters.pkl')

