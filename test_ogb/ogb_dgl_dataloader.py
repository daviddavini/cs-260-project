# -*- coding: utf-8 -*-
"""
Data loader to look at the obgn-arxiv dataset for CS 260 project

Created on Tue Feb  9 17:32:13 2021

@author: Grace Li
"""

# To use this data loader, you need to have
# ogb installed, directions here: https://ogb.stanford.edu/docs/home/
# dgl installed, directions here: https://www.dgl.ai/pages/start.html

# About the dataset:
# There are 163,343 nodes in the dataset representing arxiv papers in CS
# The training set is papers published in 2017
# The validation set is papers published in 2018
# The test set is papers published since the start of 2019
# The labels are 0-40 representing the paper category. 
# The categories are found here and I think are in the same order:
#    https://arxiv.org/category_taxonomy
# Each node in the graph (it's a dgl graph object) has attached 
# a 128-sized vector 'feat' from NLP of abstract/title and 'year' when published

import numpy as np
import torch as th
from matplotlib import pyplot as plt
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

#Load dataset
dataset = DglNodePropPredDataset(name = 'ogbn-arxiv')

#Get training, validation and test set indicies
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

#Get the graph and labels
G, label = dataset[0] # G: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
num_classes = dataset.num_classes

#Get the feature vectors for each data set
train_feat = G.ndata['feat'][train_idx]
valid_feat = G.ndata['feat'][valid_idx]
test_feat = G.ndata['feat'][test_idx]


# %%Get frequencies of each paper category, and plot the histogram colored by train, validate, and test sets

# The histogram shows that the paper categories are very imbalanced. The top 3 categories: cs.GL (graphics), 
# cs.MA (multiagent systems) and cs.NE (neural and evolutionary computing) have many more papers >21000.
# CS.DM (discrete math), cs.SC (symbolic computation), cs.IT (information theory) and some others have very few <1000.

category_names = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
                  "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
                  "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
                  "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
                  "cs.SD", "cs.SE", "cs.SI", "cs.SY"]

train_labels = th.flatten(label[train_idx]).numpy()
valid_labels = th.flatten(label[valid_idx]).numpy()
test_labels = th.flatten(label[test_idx]).numpy()

train_freq, valid_freq, test_freq = [], [], []

for i in range(num_classes):
    train_freq.append(np.count_nonzero(train_labels==i))
    valid_freq.append(np.count_nonzero(valid_labels==i))
    test_freq.append(np.count_nonzero(test_labels==i))

train_freq, valid_freq, test_freq = np.array(train_freq), np.array(valid_freq), np.array(test_freq)

# Plot histogram in alphebetical order of paper categories
fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(category_names, train_freq, color = 'tab:blue')
ax.bar(category_names, valid_freq, bottom = train_freq, color = 'tab:purple')
ax.bar(category_names, test_freq, bottom = (valid_freq + train_freq), color = 'tab:red')
ax.legend(labels=['Train', 'Validate', 'Test'], prop={'size': 15})
plt.setp(ax.get_xticklabels(), rotation = 88, horizontalalignment = 'center')
ax.tick_params(axis='both', labelsize = 13)
plt.title("Distribution of Paper Categories", fontdict={'fontsize':25})
plt.ylabel('Frequency', fontdict={'fontsize':15})
plt.savefig('class_histogram.png',bbox_inches='tight')
plt.show()

# Plot histogram in frequency order of paper categories for training set
ordering = np.argsort(np.array(train_freq))
sorted_train_freq = np.sort(train_freq)   
sorted_valid_freq = valid_freq[ordering]
sorted_test_freq = test_freq[ordering]
sorted_names = []
for i in ordering:
    sorted_names.append(category_names[i])

fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(sorted_names, sorted_train_freq, color = 'tab:blue')
ax.bar(sorted_names, sorted_valid_freq, bottom = sorted_train_freq, color = 'tab:purple')
ax.bar(sorted_names, sorted_test_freq, bottom = (sorted_valid_freq + sorted_train_freq), color = 'tab:red')
ax.legend(labels=['Train', 'Validate', 'Test'], prop={'size': 15})
plt.setp(ax.get_xticklabels(), rotation = 60, horizontalalignment = 'center')
ax.tick_params(axis='both', labelsize = 13)
plt.title("Distribution of Paper Categories", fontdict={'fontsize':25})
plt.ylabel('Frequency', fontdict={'fontsize':15})
plt.savefig('class_histogram_sorted.png',bbox_inches='tight')
plt.show()

