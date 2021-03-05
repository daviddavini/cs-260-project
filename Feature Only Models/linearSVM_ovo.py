# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:13:55 2021

@author: glifi
"""

#Linear SVM with one-vs-one approach

import math
import time
import numpy as np  

from scipy import io
from sklearn import metrics
import itertools
from matplotlib import pyplot as plt

import torch as th
from sklearn.svm import SVC, LinearSVC
import torch.utils.data as td
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


foldername = 'linearSVM ovo'

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
    #return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]
    return evaluator.eval({"y_pred": pred, "y_true": labels})["acc"]

# %% Linear SVM with One vs One Classification

print("Starting One-Vs-One Linear SVM (SVC)")
tic = time.time()

svm = SVC(kernel='linear', C=1)
svm.fit(features[train_idx], labels[train_idx])

print("Train time: ", time.time() - tic)


print("Generating predictions")
tic = time.time()

pred = svm.predict(features)

print("Prediction time: ", time.time() - tic)

# %%accuracy

dict = {'pred':pred, 'labels':labels}
io.savemat(foldername+"/predictions.mat", dict)

pred2D = np.reshape(pred, (pred.shape[0],1))
labels2D = np.reshape(labels, (labels.shape[0],1))

print("Calculating Accuracy")
tic = time.time()

train_acc = compute_acc(pred2D[train_idx], labels2D[train_idx], evaluator)
val_acc = compute_acc(pred2D[val_idx], labels2D[val_idx], evaluator)
test_acc = compute_acc(pred2D[test_idx], labels2D[test_idx], evaluator)

print("Calculation time: ", time.time() - tic)

print("Train Accuracy: ", train_acc)
print("Val Accuracy:", val_acc)
print("Test Accuracy: ", test_acc)

# %% Generate histogram of predicted labels

category_names = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
                  "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
                  "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
                  "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
                  "cs.SD", "cs.SE", "cs.SI", "cs.SY"]


# Split predicted cateogories by train, validate and test sets
train_pred = pred[train_idx]
val_pred = pred[val_idx]
test_pred = pred[test_idx]

# Get the ground truth labels for train set for sorting order later
train_labels = labels[train_idx]

true_train_freq, train_freq, val_freq, test_freq = [], [], [], []

for i in range(n_classes):
    true_train_freq.append(np.count_nonzero(train_labels==i))
    train_freq.append(np.count_nonzero(train_pred==i))
    val_freq.append(np.count_nonzero(val_pred==i))
    test_freq.append(np.count_nonzero(test_pred==i))

train_freq, val_freq, test_freq = np.array(train_freq), np.array(val_freq), np.array(test_freq)

# Plot histogram in alphebetical order of paper categories
fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(category_names, train_freq, color = 'tab:blue')
ax.bar(category_names, val_freq, bottom = train_freq, color = 'tab:purple')
ax.bar(category_names, test_freq, bottom = (val_freq + train_freq), color = 'tab:red')
ax.legend(labels=['Train', 'Validate', 'Test'], prop={'size': 15})
plt.setp(ax.get_xticklabels(), rotation = 90, horizontalalignment = 'center')
ax.tick_params(axis='both', labelsize = 13)
plt.title("Distribution of Predicted Paper Categories", fontdict={'fontsize':25})
plt.ylabel('Frequency', fontdict={'fontsize':15})
plt.savefig(foldername + "/pred_class_histogram.png",bbox_inches='tight')
plt.show()

# Plot histogram in frequency order of ground truth paper categories for training set

ordering = np.argsort(np.array(true_train_freq))
sorted_train_freq = train_freq[ordering] 
sorted_val_freq = val_freq[ordering]
sorted_test_freq = test_freq[ordering]
sorted_names = []
for i in ordering:
    sorted_names.append(category_names[i])

fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(sorted_names, sorted_train_freq, color = 'tab:blue')
ax.bar(sorted_names, sorted_val_freq, bottom = sorted_train_freq, color = 'tab:purple')
ax.bar(sorted_names, sorted_test_freq, bottom = (sorted_val_freq + sorted_train_freq), color = 'tab:red')
ax.legend(labels=['Train', 'Validate', 'Test'], prop={'size': 15})
plt.setp(ax.get_xticklabels(), rotation = 90, horizontalalignment = 'center')
ax.tick_params(axis='both', labelsize = 13)
plt.title("Distribution of Predicted Paper Categories", fontdict={'fontsize':25})
plt.ylabel('Frequency', fontdict={'fontsize':15})
plt.savefig(foldername + "/pred_class_histogram_sorted.png",bbox_inches='tight')
plt.show()

# %% Plot the contigency matrices

# Function for plotting the confusion matrix. Borrowed from ECE 219 project 2
def plot_mat(mat, xticklabels = None, yticklabels = None, pic_fname = None, size=(-1,-1), if_show_values = True, 
             num_decimals = 0, colorbar = True, grid = 'k', xlabel = None, ylabel = None, title = None, 
             vmin=None, vmax=None, fontsize = {'title':15, 'axislabel': 15, 'small': 10}):
    if size == (-1, -1):
        size = (mat.shape[1] / 3, mat.shape[0] / 3)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1,1,1)

    # im = ax.imshow(mat, cmap=plt.cm.Blues)
    im = ax.pcolor(mat, cmap=plt.cm.Blues, linestyle='-', linewidth=0.5, edgecolor=grid, vmin=vmin, vmax=vmax)
    
    if colorbar:
        cbar = plt.colorbar(im, aspect = 30) #fraction=0.046, pad=0.07)
        cbar.ax.tick_params(labelsize=fontsize['axislabel']) 
    # tick_marks = np.arange(len(classes))
    # Ticks
    lda_num_topics = mat.shape[0]
    nmf_num_topics = mat.shape[1]
    yticks = np.arange(lda_num_topics)
    xticks = np.arange(nmf_num_topics)
    ax.set_xticks(xticks + 0.5)
    ax.set_yticks(yticks + 0.5)
    if xticklabels:
        ax.tick_params(axis='x', labelrotation = 90)
    if xticklabels is None:
        xticklabels = [str(i) for i in xticks]
    if yticklabels is None:
        yticklabels = [str(i) for i in yticks]
    ax.set_xticklabels(xticklabels, fontsize = fontsize['small'])
    ax.set_yticklabels(yticklabels, fontsize = fontsize['small'])

    # Minor ticks
    # ax.set_xticks(xticks, minor=True);
    # ax.set_yticks(yticks, minor=True);
    # ax.set_xticklabels([], minor=True)
    # ax.set_yticklabels([], minor=True)

    # ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)

    # tick labels on left, right and bottom
    ax.tick_params(labelright = True, labeltop = False)

    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize['axislabel'])
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize['axislabel'])
    if title:
        plt.title(title, fontsize=fontsize['title'])

    # im = ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.invert_yaxis()

    # thresh = mat.max() / 2

    def show_values(pc, fmt="%." + str(num_decimals) + "f", **kw):
        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in itertools.zip_longest(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw, fontsize=fontsize['small'])
    if if_show_values:
        show_values(im)
    # for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
    #     ax.text(j, i, "{:.2f}".format(mat[i, j]), fontsize = 4,
    #              horizontalalignment="center",
    #              color="white" if mat[i, j] > thresh else "black")

    plt.tight_layout()
    if pic_fname:
        plt.savefig(pic_fname, dpi=200, facecolor='w', bbox_inches='tight')
    plt.show()

#sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#We normalize against the true labels, so each matrix entry is divided by its row sum

# Get the ground truth labels
# From the histogram plot, the predicted labels are already in train_pred, val_pred, and test_pred
train_labels = labels[train_idx]
val_labels = labels[val_idx]
test_labels = labels[test_idx]

# Generate the contingency matrix for training set
train_matrix = metrics.confusion_matrix(train_labels, train_pred, normalize='true')
pic_fname = foldername + "/train_matrix.png"
plot_mat(train_matrix,xlabel='Cluster Class', ylabel='Actual Class', title='Normalized Confusion Matrix', num_decimals = 2,
          xticklabels = category_names, yticklabels = category_names,
          size=(35,30), fontsize = {'title':35, 'axislabel':25, 'small':15}, pic_fname = pic_fname)

# Generate the contingency matrix for valation set
val_matrix = metrics.confusion_matrix(val_labels, val_pred, normalize='true')
pic_fname = foldername + "/val_matrix.png"
plot_mat(val_matrix, xlabel='Cluster Class', ylabel='Actual Class', title='Normalized Confusion Matrix', num_decimals = 2,
          xticklabels = category_names, yticklabels = category_names,
          size=(35,30), fontsize = {'title':35, 'axislabel':25, 'small':15}, pic_fname = pic_fname)

# Generate the contingency matrix for test set
test_matrix = metrics.confusion_matrix(test_labels, test_pred, normalize='true')
pic_fname = foldername + "/test_matrix.png"
plot_mat(test_matrix, xlabel='Cluster Class', ylabel='Actual Class', title='Normalized Confusion Matrix', num_decimals = 2,
         xticklabels = category_names, yticklabels = category_names,
         size=(35,30), fontsize = {'title':35, 'axislabel':25, 'small':15}, pic_fname = pic_fname)