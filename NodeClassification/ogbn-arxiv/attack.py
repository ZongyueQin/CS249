import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, PGDAttack, MinMax
from MyDeepRobustAttack import DICE, Random
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
