import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import DataLoader 
import numpy as np
import argparse

global_args = None

def my_kld(inputs, targets):
    # assume neighbor p or q are softmax-ed
    log_inputs = F.log_softmax(inputs, 1)
    target_prob = F.softmax(targets, 1)
    loss = nn.KLDivLoss().to(global_args.device)
    return loss(log_inputs, target_prob)
    

def my_neighbor_kld(neighbor_outputs, adv_output):
    loss = 0
    for i in range(len(neighbor_outputs)):
        loss += my_kld(adv_output, neighbor_outputs[i])
    return loss

def EdgeIndex2NeighborLists(edge_index, node_num, direct = False):
    edge_num = edge_index.size()[1]
    result = [[] for i in range(node_num)]
    for i in range(edge_num):
        f_node = edge_index[0,i].item()
        t_node = edge_index[1,i].item()
        result[f_node].append(t_node)
        if direct == False:
            result[t_node].append(f_node)
    for i, nlist in enumerate(result):
        if len(nlist) == 0:
            nlist.append(i)
    return result

"""
sample #global_args.num_neighbor neighbors for each node
neighbor_lists is a list of lists which contains each node's neighbors
"""
def sample_neighbors(neighbor_lists):
    results = []
    for node in range(len(neighbor_lists)):
        neighbors = np.random.choice(neighbor_lists[node], size = (global_args.num_neighbors))
        results.append(neighbors)
    return np.array(results)

def generate_GraphAT_perturbation(x, edge_index, model, neighbor_outputs, mask, *args, **kwargs):
    model.eval()
    x_copy = Variable(x, requires_grad = True).to(global_args.device)
    node_output = model(x_copy, edge_index, *args, **kwargs)[mask]
    loss = my_neighbor_kld(neighbor_outputs, node_output)
    grad = torch.autograd.grad(loss, [x_copy])[0]
    d = Variable(grad, requires_grad = False).to(global_args.device)
    return global_args.epsilon_graph * F.normalize(d, p=2)

"""
x:              node features
adj:            adjacency matrix
model:          target model
node_output:    output representation of each node
neighbor_ids:   ids of sampled neighbors of each training node
num_neighbors:  number of sampled neighbors for each training node
train_mask:     mask of training nodes
"""
def GraphAT_Loss(x, edge_index, model, node_output, neighbor_ids, mask, *args, **kwargs):

    neighbor_outputs = list()
    for i in range(global_args.num_neighbors):
        
        neighbor_output = node_output[neighbor_ids[:,i]]
        neighbor_output = neighbor_output[mask,:]
        neighbor_output = neighbor_output.detach()
        neighbor_outputs.append(neighbor_output)
    
    r_gadv = generate_GraphAT_perturbation(x, edge_index, model, neighbor_outputs, mask, *args, **kwargs)
    
    model.train()
    adv_output = model(x+r_gadv, edge_index, *args, **kwargs)
    adv_output = adv_output[mask]
    gat_loss = my_neighbor_kld(neighbor_outputs, adv_output)
    return gat_loss, adv_output
    
def generate_virtual_adversarial_perturbation(x, edge_index, model, node_output, *args, **kwargs):
    d = torch.randn(x.size())
    for _ in range(global_args.num_power_iterations):
        d = global_args.xi * F.normalize(d, p = 2)    
        d = Variable(d, requires_grad = True).to(global_args.device)
        model.eval()
        adv_output = model(x+d, edge_index, *args, **kwargs)
        loss = my_kld(node_output, adv_output)
        grad = torch.autograd.grad(loss, [d])[0]
        d = Variable(grad, requires_grad = False).to(global_args.device)
    return global_args.epsilon * F.normalize(d, p=2)

def GraphVAT_Loss(x, edge_index, model, node_output, *args, **kwargs):
    r_vadv = generate_virtual_adversarial_perturbation(x, edge_index, model, node_output, *args, **kwargs)
    model.train()
    adv_output =model(x+r_vadv, edge_index, *args, **kwargs)
    vat_loss = my_kld(node_output, adv_output)
    return vat_loss
