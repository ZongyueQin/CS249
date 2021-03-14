# import numpy as np
# from deeprobust.graph.data import Dataset
# from deeprobust.graph.defense import GCN
# from deeprobust.graph.defense.adv_training import AdvTraining
import torch
import os
import argparse
from attack import apply_Random, apply_DICE, apply_PGDAttack
from deeprobust.graph.data import PrePtbDataset, Dataset
from deeprobust.graph.defense import GCNJaccard, GCNSVD
from MyDeepRobustRGCN import RGCN

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.dataset = 'pubmed'
#args.n_classes = 10
args.lr = 1e-2
args.n_hids = 32
args.n_heads = 1                                 # currently not used in GNN2 implementation?
args.n_layer = 2                                 # same in default
args.dropout = 0.5
args.num_epochs = 200                            # same in default
args.weight_decay = 0.01
args.w_robust = 0
args.step_per_epoch = 1
args.device = device
args.n_perturbations = [0.01, 0.02, 0.04]
args.max_no_increase_epoch_num = 10

def fit_model(name, model, features, adj, labels, idx_train, idx_val, epochs):
    if name == 'jaccard':
        model.fit(features, adj, labels, idx_train, idx_val, threshold=0.03, train_iters=epochs)  # jaccard
    if name == 'svd':
        model.fit(features, adj, labels, idx_train, idx_val, k=20, train_iters=epochs)            # svd
    if name == 'rgcn':
        model.fit(features, adj, labels, idx_train, idx_val, train_iters=epochs)

def main():
    data = Dataset(root='dataset/', name=args.dataset, seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    num_edges = 44338
    train_idx = idx_train
    valid_idx = idx_val
    test_idx = idx_test

    jaccard = GCNJaccard(nfeat=features.shape[1],
                          nhid=args.n_hids,
                          nclass=labels.max().item() + 1,
                          dropout=args.dropout,
                          lr=args.lr, weight_decay=args.weight_decay,
                          device=device).to(device)
    jaccard_2 = GCNJaccard(nfeat=features.shape[1],
                          nhid=args.n_hids,
                          nclass=labels.max().item() + 1,
                          dropout=args.dropout,
                          lr=args.lr, weight_decay=args.weight_decay,
                          device=device).to(device)

    print("This is GCN Jaccard \n\n\n")

    svd = GCNSVD(nfeat=features.shape[1],
                        nhid=args.n_hids,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout,
                        lr = args.lr,
                        weight_decay = args.weight_decay,
                        device=device).to(device)
    svd_2 = GCNSVD(nfeat=features.shape[1],
                        nhid=args.n_hids,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout,
                        lr = args.lr,
                        weight_decay = args.weight_decay,
                        device=device).to(device)


    print("This is GCN SVD \n\n")

    rgcn = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1],
                   nclass=labels.max() + 1, nhids=[args.n_hids],
                   dropout=args.dropout,
                   lr=args.lr,
                   device=device).to(device)
    rgcn_2 = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1],
                   nclass=labels.max() + 1, nhids=[args.n_hids],
                   dropout=args.dropout,
                   lr=args.lr,
                   device=device).to(device)
    
    print("This is RGCN \n\n")


    print("clean!! \n")
    models = [('jaccard', jaccard, jaccard_2), ('svd', svd, svd_2), ('rgcn', rgcn, rgcn_2)]
    #models = [('jaccard', jaccard, jaccard_2), ('svd', svd, svd_2)]
    for name, model, _ in models:
        print(name)
        fit_model(name, model, features, adj, labels, idx_train, idx_val, args.num_epochs*args.step_per_epoch)
        model.test(idx_test)

    print("random!! \n")
    y_test = labels[test_idx]
    y_test = torch.from_numpy(y_test).long().to(device)
   
    for ratio in args.n_perturbations:
        perturbed_adj = apply_Random(adj, n_perturbations = int(ratio*num_edges))
        for name, model, model_2 in models:
            print(name)
            predict = torch.argmax(model.predict(features = features, adj = perturbed_adj), dim=1)
            predict = predict[test_idx]
            correct = (predict.view(-1).long()==y_test).sum()
            total = len(test_idx)
            acc = correct.float() / float(total)
            print('Test Robust Accuracy (Random, %.2f): %.3f' % (ratio, acc))
            fit_model(name, model_2, features, adj, labels, idx_train, idx_val, args.num_epochs*args.step_per_epoch)
            print('posinoning atttack')
            model_2.test(test_idx)

    """ Robust Accuracy (DICE) """
    for ratio in args.n_perturbations:
        perturbed_adj = apply_DICE(adj, labels, n_perturbations = int(ratio*num_edges))
        for name, model, model_2 in models:
            print(name)
            predict = torch.argmax(model.predict(features = features, adj = perturbed_adj), dim=1)
            predict = predict[test_idx]
            correct = (predict.view(-1).long()==y_test).sum()
            total = len(test_idx)
            acc = correct.float() / float(total)
            print('Test Robust Accuracy (Random, %.2f): %.3f' % (ratio, acc))
            fit_model(name, model_2, features, adj, labels, idx_train, idx_val, args.num_epochs*args.step_per_epoch)
            print('posinoning atttack')
            model_2.test(test_idx)


    print('meta')
    for rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
        perturbed_data = PrePtbDataset(root = 'dataset/',
                                           name = args.dataset,
                                           attack_method = 'meta',
                                           ptb_rate = rate)
        perturbed_adj = perturbed_data.adj
        for name, model, model_2 in models:
            print(name)
            predict = torch.argmax(model.predict(features = features, adj = perturbed_adj), dim=1)
            predict = predict[test_idx]
            correct = (predict.view(-1).long()==y_test).sum()
            total = len(test_idx)
            acc = correct.float() / float(total)
            print('Test Robust Accuracy (Random, %.2f): %.3f' % (ratio, acc))
            fit_model(name, model_2, features, adj, labels, idx_train, idx_val, args.num_epochs*args.step_per_epoch)
            print('posinoning atttack')
            model_2.test(test_idx)


    print('nettack')
    for rate in [1.0, 2.0, 3.0, 4.0, 5.0]:
        perturbed_data = PrePtbDataset(root = 'dataset/',
                                           name = args.dataset,
                                           attack_method = 'nettack',
                                           ptb_rate = rate)
        # nettack_adj_list.append(perturbed_data.adj)
        perturbed_adj = perturbed_data.adj
        for name, model, model_2 in models:
            print(name)
            predict = torch.argmax(model.predict(features = features, adj = perturbed_adj), dim=1)
            predict = predict[test_idx]
            correct = (predict.view(-1).long()==y_test).sum()
            total = len(test_idx)
            acc = correct.float() / float(total)
            print('Test Robust Accuracy (Random, %.2f): %.3f' % (ratio, acc))
            fit_model(name, model_2, features, adj, labels, idx_train, idx_val, args.num_epochs*args.step_per_epoch)
            print('posinoning atttack')
            model_2.test(test_idx)




    """
    print("meta! \n")
    for adj in meta_adj_list:
        # model.fit(features, adj, labels, idx_train, idx_val, threshold=0.03, train_iters=args.num_epochs)    # jaccard
        model.fit(features, adj, labels, idx_train, idx_val, k=20, train_iters=args.num_epochs)
        # model.fit(features, adj, labels, idx_train, idx_val, train_iters=args.num_epochs)
        model.test(idx_test)

    print("dice!! \n")
    # model.fit(features, perturbed_adj_dice, labels, idx_train, idx_val, threshold=0.03, train_iters=args.num_epochs)
    model.fit(features, perturbed_adj_dice, labels, idx_train, idx_val, k=20, train_iters=args.num_epochs)
    # model.fit(features, perturbed_adj_dice, labels, idx_train, idx_val,train_iters=args.num_epochs)
    model.test(idx_test)


    print("net!! \n")
    for adj in nettack_adj_list:
        # model.fit(features, adj, labels, idx_train, idx_val, threshold=0.03, train_iters=args.num_epochs)
        model.fit(features, adj, labels, idx_train, idx_val, k=20, train_iters=args.num_epochs)
        # model.fit(features, adj, labels, idx_train, idx_val, train_iters=args.num_epochs)
        model.test(idx_test)
    """



if __name__ == "__main__":
    main()
