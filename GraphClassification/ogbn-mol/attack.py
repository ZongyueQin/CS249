import numpy as np
import scipy.sparse as sp
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, PGDAttack, MinMax
from MyDeepRobustAttack import DICE, Random
from torch_geometric.data import Data
import torch
import os

def apply_Metattack(model, features, adj, labels, idx_train, idx_unlabeled,
                    #attack_structure = True,
                    #attack_features = False,
                    device = 'cpu',
                    lambda_ = 0,
                    n_perturbations = 10,
                    ll_constraint = False):
  attack_model = Metattack(model, nnodes = adj.shape[0],
                         feature_shape = features.shape,
                         attack_structure = True,
                         attack_features = False,
                         device = device, lambda_=lambda_).to(device)

  attack_model.attack(features, adj, labels, idx_train,
                      idx_unlabeled, n_perturbations=n_perturbations,
                      ll_constraint=ll_constraint)
  return attack_model.modified_adj

def apply_Random(adj, n_perturbations = 10):
  attack_model = Random()
  attack_model.attack(adj, n_perturbations = n_perturbations)
  return attack_model.modified_adj

def apply_DICE(adj, node_labels, n_perturbations = 10):
  attack_model = DICE()
  attack_model.attack(adj, labels = node_labels, n_perturbations = n_perturbations)
  return attack_model.modified_adj

def pyg_apply_Random(data, ptb_ratio):
  n = data.num_nodes
  num_edges = data.edge_index.shape[1]
  n_perturbations = int(ptb_ratio*num_edges)
  adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), 
            (data.edge_index[0], data.edge_index[1])), shape=(n,n))
  new_adj = apply_Random(adj, n_perturbations)
  new_data = Data(x = data.x, batch=data.batch, y=data.y) 
  new_data.edge_index = torch.LongTensor(new_adj.nonzero())
  new_data.edge_attr = torch.zeros([new_data.edge_index.shape[1], 1], dtype=torch.long)
  return new_data

def pyg_apply_DICE(data, ptb_ratio):
  raise RuntimeError('DICE unavailable')
  n = data.num_nodes
  num_edges = data.edge_index.shape[1]
  node_labels = data.y.numpy()
  n_perturbations = int(ptb_ratio*num_edges)
  adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), 
            (data.edge_index[0], data.edge_index[1])), shape=(n,n))
  new_adj = apply_DICE(adj, node_labels, n_perturbations)
  new_data = Data(x = data.x, batch=data.batch)
  new_data.edge_index = torch.LongTensor(new_adj.nonzero())
  return new_data



def apply_PGDAttack(model, features, adj, labels, idx_train,
                    #attack_structure = True,
                    #attack_features = False,
                    device = 'cpu',
                    n_perturbations = 10,
                    loss_type = 'CE'):
  attack_model = PGDAttack(model, nnodes = adj.shape[0],
                         loss_type = loss_type,
                         device = device).to(device)

  attack_model.attack(features, adj.toarray(), labels, idx_train,
                      n_perturbations= n_perturbations)

  return attack_model.modified_adj

def apply_MinMax(model, features, adj, labels, idx_train,
                    #attack_structure = True,
                    #attack_features = False,
                    device = 'cpu',
                    n_perturbations = 10,
                    loss_type = 'CE'):
  attack_model = MinMax(model, nnodes = adj.shape[0],
                         loss_type = loss_type,
                         device = device).to(device)

  attack_model.attack(features, adj, labels, idx_train,
                      n_perturbations= n_perturbations)

  return attack_model.modified_adj
