# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:42:22 2021

@author: glifi
"""

## Naive-Bayes classifier is not suitable due to the negative values in the feature vectors.


import math
import time
import numpy as np

from scipy import io
from sklearn import metrics
import itertools
from matplotlib import pyplot as plt

import torch as th
from sklearn.naive_bayes import MultinomialNB, ComplementNB
import torch.utils.data as td
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


foldername = 'naive bayes'

# %% Load dataset
dataset = DglNodePropPredDataset(name = 'ogbn-arxiv')

#Get training, validation and test set indicies
split_idx = dataset.get_idx_split()
train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

#Get the graph, labels, and feature vectors
graph, labels = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
n_classes = dataset.num_classes #num_classes should be 40
features = graph.ndata['feat'] #feature vectors
features = features.numpy() #convert to numpy matrix
labels = th.flatten(labels).numpy() #convert to numpy vector

#Load evaluator
evaluator = Evaluator(name="ogbn-arxiv")

# Define accuracy function
def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]

# %% Multinomial Naive Bayes

print("Starting Multinomial Naive Bayes")
tic = time.time()

multinomialNB = MultinomialNB()
multinomialNB.fit(features[train_idx], labels[train_idx])

toc = time.time()
print("Train time: ", toc - tic)

pred = multinomialNB.predict(features)

train_acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)
val_acc = compute_acc(pred[val_idx], labels[val_idx], evaluator)
test_acc = compute_acc(pred[test_idx], labels[test_idx], evaluator)

print("Train Accuracy: ", train_acc)
print("Val Accuracy:", val_acc)
print("Test Accuracy: ", test_acc)
