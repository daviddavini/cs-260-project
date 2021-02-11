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


# %%Get frequencies of each paper category, and plot the histogram
label_array = th.flatten(label)
label_array = label.numpy()

category_names = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
                  "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
                  "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
                  "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
                  "cs.SD", "cs.SE", "cs.SI", "cs.SY"]

frequency=[]
for i in range(num_classes):
    frequency.append(np.count_nonzero(label_array==i))
    
fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(category_names,frequency)
plt.setp(ax.get_xticklabels(), rotation = 60, horizontalalignment = 'center')
ax.tick_params(axis='both', labelsize = 13)
plt.title("Distribution of Paper Categories", fontdict={'fontsize':20})
plt.ylabel('Frequency', fontdict={'fontsize':15})
plt.savefig('Class_histogram.png',bbox_inches='tight')
plt.show()

# The histogram shows that the paper categories are very imbalanced. The top 3 categories: cs.GL (graphics), 
# cs.MA (multiagent systems) and cs.NE (neural and evolutionary computing) have many more papers >21000.
# CS.DM (discrete math), cs.SC (symbolic computation), cs.IT (information theory) and some others have very few <1000.