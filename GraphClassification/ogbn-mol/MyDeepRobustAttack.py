import random
import numpy as np
import scipy.sparse as sp
from deeprobust.graph.global_attack import BaseAttack

class Random(BaseAttack):
    """ Randomly adding edges to the input graph

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import Random
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = Random()
    >>> model.attack(adj, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Random, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        assert not self.attack_features, 'RND does NOT support attacking features'

    def attack(self, ori_adj, n_perturbations, type='add', **kwargs):
        """Generate attacks on the input graph.

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of edge removals/additions.
        type: str
            perturbation type. Could be 'add', 'remove' or 'flip'.

        Returns
        -------
        None.

        """

        if self.attack_structure:
            modified_adj = self.perturb_adj(ori_adj, n_perturbations, type)
            self.modified_adj = modified_adj


    def perturb_adj(self, adj, n_perturbations, type='add'):
        """Randomly add, remove or flip edges.

        Parameters
        ----------
        adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of edge removals/additions.
        type: str
            perturbation type. Could be 'add', 'remove' or 'flip'.

        Returns
        ------
        scipy.sparse matrix
            perturbed adjacency matrix
        """
        # adj: sp.csr_matrix
        modified_adj = adj.tolil()

        type = type.lower()
        assert type in ['add', 'remove', 'flip']

        if type == 'flip':
            # sample edges to flip
            edges = self.random_sample_edges(adj, n_perturbations, exclude=set())
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
                modified_adj[n2, n1] = 1 - modified_adj[n2, n1]

        if type == 'add':
            # sample edges to add
            nonzero = set(zip(*adj.nonzero()))
            edges = self.random_sample_edges(adj, n_perturbations, exclude=nonzero)
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1
                modified_adj[n2, n1] = 1

        if type == 'remove':
            # sample edges to remove
            nonzero = np.array(sp.triu(adj, k=1).nonzero()).T
            indices = np.random.permutation(nonzero)[: n_perturbations].T
            modified_adj[indices[0], indices[1]] = 0
            modified_adj[indices[1], indices[0]] = 0

        return modified_adj


    def perturb_features(self, features, n_perturbations):
        """Randomly perturb features.
        """
        raise NotImplementedError
        print('number of pertubations: %s' % n_perturbations)
        return modified_features


    def inject_nodes(self, adj, n_add, n_perturbations):
        """For each added node, randomly connect with other nodes.
        """
        # adj: sp.csr_matrix
        # TODO
        print('number of pertubations: %s' % n_perturbations)
        raise NotImplementedError

        modified_adj = adj.tolil()
        return modified_adj


    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            # t = tuple(random.sample(range(0, adj.shape[0]), 2))
            t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

class DICE(BaseAttack):
    """As is described in ADVERSARIAL ATTACKS ON GRAPH NEURAL NETWORKS VIA META LEARNING (ICLR'19),
    'DICE (delete internally, connect externally) is a baseline where, for each perturbation,
    we randomly choose whether to insert or remove an edge. Edges are only removed between
    nodes from the same classes, and only inserted between nodes from different classes.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'


    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import DICE
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = DICE()
    >>> model.attack(adj, labels, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(DICE, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        assert not self.attack_features, 'DICE does NOT support attacking features'

    def attack(self, ori_adj, labels, n_perturbations, **kwargs):
        """Delete internally, connect externally. This baseline has all true class labels
        (train and test) available.

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        labels:
            node labels
        n_perturbations : int
            Number of edge removals/additions.

        Returns
        -------
        None.

        """

        # ori_adj: sp.csr_matrix

        print('number of pertubations: %s' % n_perturbations)
        modified_adj = ori_adj.tolil()

        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)

        nonzero = set(zip(*ori_adj.nonzero()))
        indices = sp.triu(modified_adj).nonzero()
        possible_indices = [x for x in zip(indices[0], indices[1])
                            if labels[x[0]] == labels[x[1]]]

        remove_indices = np.random.permutation(possible_indices)[: n_remove]
        modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0
        modified_adj[remove_indices[:, 1], remove_indices[:, 0]] = 0

        n_insert = n_perturbations - n_remove

        # # sample edges to add
        # nonzero = nonzero
        # edges = self.random_sample_edges(adj, n_insert, exclude=nonzero)
        # for n1, n2 in edges:
        #     modified_adj[n1, n2] += 1
        #     modified_adj[n2, n1] += 1

        # sample edges to add
        for i in range(n_insert):
            # select a node
            node1 = np.random.randint(ori_adj.shape[0])
            possible_nodes = [x for x in range(ori_adj.shape[0])
                              if labels[x] != labels[node1] and modified_adj[x, node1] == 0]
            # select another node
            node2 = possible_nodes[np.random.randint(len(possible_nodes))]
            modified_adj[node1, node2] = 1
            modified_adj[node2, node1] = 1

        #self.check_adj(modified_adj)
        self.modified_adj = modified_adj



    def sample_forever(self, adj, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))


    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]
