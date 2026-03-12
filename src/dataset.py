import torch
import pickle
import random as rd
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter


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
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()


def load_dataset(matrix, adjlist):
	"""
	Load dataset from matrix and adjacency list files.
	:param matrix: the filename of the matrix file
	:param adjlist: the filename of the adjacency list file
	:return: labels, features, and homogenous adjacency list
	"""
	data_file = loadmat(matrix)

	labels = data_file['label'].flatten()
	features = data_file['features'].todense().A

	yelp_homo = data_file['homo'] #C
	sparse_to_adjlist(yelp_homo, adjlist) #C

	# load the preprocessed adj_lists
	with open(adjlist, 'rb') as file:
		homogenous = pickle.load(file)
	file.close()
	return labels, features, homogenous


def split_dataset(features, labels, test_size=0.2, random_state=99):

	# split data into test and train
	xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = test_size,
																		stratify=labels,
																		random_state = random_state)
	return xtrain, xtest, ytrain, ytest

