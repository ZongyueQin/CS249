import os
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from preprocess import GCNSVD, GCNJaccard
import GraphAT
from GraphAT import GraphAT_Loss, GraphVAT_Loss, sample_neighbors, EdgeIndex2NeighborLists
from attack import pyg_apply_Random, pyg_apply_DICE

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

perturbation_ratios = [0.01, 0.02, 0.04]
graph_vat_w = 0.5
graph_at_w = 0.5

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            # graphAT
            if graph_vat_w > 0 or graph_at_w > 0:
                ori_out = model(batch, node_level=True)
            if graph_at_w > 0:
                neighbor_lists = EdgeIndex2NeighborLists(batch.edge_index,batch.num_nodes, batch.is_directed())
                neighbor_ids = sample_neighbors(neighbor_lists)
                loss_gat, _ = GraphAT_Loss(batch, model, ori_out,
                                    neighbor_ids, None, node_level=True)
                loss += graph_at_w * loss_gat
            if graph_vat_w > 0:
                loss_gvat = GraphVAT_Loss(batch, model, ori_out)
                loss += graph_vat_w * loss_gvat
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator, adv_mode = None, *args, **kwargs):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if adv_mode == 'random':
            batch = pyg_apply_Random(batch, *args, **kwargs)
        elif adv_mode == 'DICE':
            batch = pyg_apply_DICE(batch, *args, **kwargs)

        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')

    parser.add_argument('--jaccard', type=float, default=0, 
                        help='a value between 0 and 1, 0 if not preprocess with jaccard, otherwise it is jaccard threshold')
    args = parser.parse_args()
    GraphAT.global_args = args
    GraphAT.global_args.num_power_iterations = 1
    GraphAT.global_args.xi = 1e-6
    GraphAT.global_args.epsilon = 1.0
    GraphAT.global_args.epsilon_graph = 0.01
    GraphAT.global_args.num_neighbors = 2

    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    dataset.data.edge_attr = dataset.data.edge_attr[:,0:1] * 0

    split_idx = dataset.get_idx_split()


    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, use_jaccard = (args.jaccard>0), jaccard_thres=args.jaccard, device=device ).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    best_valid_perf = -1

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)
        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        if valid_perf['ap'] > best_valid_perf:
            torch.save(model.state_dict(), 'model.pkl')
            best_valid_perf = valid_perf['ap']

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    print('Adverserial Testing...')
    model.load_state_dict(torch.load('model.pkl'))
    adv_perf = []
    for ratio in perturbation_ratios:
        perf = eval(model, device, test_loader, evaluator, 'random', ptb_ratio = ratio) 
        print('Random (%.2f): %.2f'%(ratio, perf['ap']))
        adv_perf.append(perf)

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train, 'ptb_ratios': perturbation_ratios, 'adv_perf': adv_perf}, args.filename)



if __name__ == "__main__":
    main()
