import logging
import torch
import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.io import loadmat
from torch_geometric.utils.convert import from_networkx

from sklearn.model_selection import train_test_split
from collections import defaultdict

logger = logging.getLogger(__name__)


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, "wb") as file:
        pickle.dump(adj_lists, file)
    file.close()


def load_dataset(matrix, adjlist):
    """
    Load dataset from matrix and adjacency list files.
    :param matrix: the filename of the matrix file
    :param adjlist: the filename of the adjacency list file
    :return: labels, features, and homogenous adjacency list
    """
    logger.info("Loading dataset from %s", matrix)
    data_file = loadmat(matrix)

    labels = data_file["label"].flatten()
    features = data_file["features"].todense().A

    n_nodes, n_features = features.shape
    n_fraud = int(labels.sum())
    logger.info("Nodes: %d | Features: %d | Fraud: %d (%.1f%%) | Legit: %d (%.1f%%)",
                n_nodes, n_features,
                n_fraud, 100 * n_fraud / n_nodes,
                n_nodes - n_fraud, 100 * (n_nodes - n_fraud) / n_nodes)

    yelp_homo = data_file["homo"]
    logger.info("Building adjacency list from homo relation (%d edges) -> %s", yelp_homo.nnz, adjlist)
    sparse_to_adjlist(yelp_homo, adjlist)

    with open(adjlist, "rb") as file:
        homogenous = pickle.load(file)
    file.close()
    logger.info("Adjacency list loaded (%d nodes)", len(homogenous))
    return labels, features, homogenous


def split_dataset(features, labels, test_size=0.2, random_state=99):
    indices = np.arange(len(features))
    xtrain, xtest, ytrain, ytest, idxtrain, idxtest = train_test_split(features,
                                                                   labels,
                                                                   indices,
                                                                   stratify=labels,
                                                                   test_size = test_size,
                                                                   random_state = random_state)
    logger.info("Split: train=%d (fraud=%.1f%%) | test=%d (fraud=%.1f%%)",
                len(xtrain), 100 * ytrain.mean(),
                len(xtest), 100 * ytest.mean())
    return (xtrain, xtest, ytrain, ytest, idxtrain, idxtest)


def graph_dataset(homogenous, features, labels, idxtrain, idxtest):
    g = nx.Graph(homogenous)
    data = from_networkx(g)
    data.x = torch.tensor(features).float()
    data.y = torch.tensor(labels)
    data.num_node_features = data.x.shape[-1]
    data.num_classes = 1 #binary classification

    A = set(range(len(labels)))
    data.train_mask = torch.tensor([x in idxtrain for x in A])
    data.test_mask = torch.tensor([x in idxtest for x in A])
    return data
