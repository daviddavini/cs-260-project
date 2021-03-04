#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import time

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from scipy import io
from sklearn import metrics
import itertools
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from models import AGNN

global device, in_feats, n_classes, epsilon

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(args):
    norm = "both" if args.use_norm else "none"

    if args.use_labels:
        model = AGNN(
            in_feats + n_classes,
            n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            attn_drop=args.attn_drop,
            norm=norm,
        )
    else:
        model = AGNN(
            in_feats,
            n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            attn_drop=args.attn_drop,
            norm=norm,
        )

    return model


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, optimizer, use_labels):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    th.nn.utils.clip_grad_norm(model.parameters(),10)
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )

def count_parameters(args):
    model = gen_model(args)
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

# %% Define class with the model arguments
class args:
    cpu = True #Run cpu only if true. This overrides the gpu value
    gpu = 0 #Change number if different GPU device ID
    n_runs = 1 #Number of model runs
    n_epochs = 1000 #2000 #Number of epochs
    use_labels = False #Use labels in the training set as input features
    use_norm = False #Use symmetrically normalized adjacency matrix
    lr = 0.002 #0.002 Learning rate
    n_layers = 2 #3 #Number of layers
    n_heads = 1 #3
    n_hidden = 256 #256  
    dropout = 0.75 #0.75
    attn_drop = 0.05
    wd = 0
    log_every = 1 #print result every log_every-th epoch
    #plot_curves = True

# Define folder to save plots and model in
foldername = "test"

# set cpu or gpu
if args.cpu:
    device = th.device("cpu")
else:
    device = th.device("cuda:%d" % args.gpu)

# load data
data = DglNodePropPredDataset(name="ogbn-arxiv")
evaluator = Evaluator(name="ogbn-arxiv")

splitted_idx = data.get_idx_split()
train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
graph, labels = data[0]

# add reverse edges
srcs, dsts = graph.all_edges()
graph.add_edges(dsts, srcs)

# add self-loop
print(f"Total edges before adding self-loop {graph.number_of_edges()}")
graph = graph.remove_self_loop().add_self_loop()
print(f"Total edges after adding self-loop {graph.number_of_edges()}")

in_feats = graph.ndata["feat"].shape[1]
n_classes = (labels.max() + 1).item()
# graph.create_format_()

train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)
labels = labels.to(device)
graph = graph.to(device)

# %% Run the model
val_accs = []
test_accs = []

# define model and optimizer
model = gen_model(args)
model = model.to(device)

optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

# training loop
total_time = 0
best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

#save accuracy and loss values
accs, train_accs, val_accs, test_accs = [], [], [], []
losses, train_losses, val_losses, test_losses = [], [], [], []

for epoch in range(1, args.n_epochs + 1):
    print("Starting Epoch ", epoch)
    
    tic = time.time()

    adjust_learning_rate(optimizer, args.lr, epoch)
    
    loss, pred = train(model, graph, labels, train_idx, optimizer, args.use_labels)
    acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

    train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
        model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator
    )
    
    toc = time.time()
    total_time += toc - tic
    
    print("Epoch run-time ", toc-tic)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_test_acc = test_acc

    if epoch % args.log_every == 0:
        print(f"\nEpoch: {epoch}/{args.n_epochs}")
        print(
            f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
            f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
            f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}"
        )

    for l, e in zip(
        [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
        [acc, train_acc, val_acc, test_acc, loss.item(), train_loss, val_loss, test_loss],
    ):
        l.append(e)

# %% Printouts

print("*" * 50)
print(f"Average epoch time: {total_time / args.n_epochs}")
print(f"Total Time: {total_time}")
print(f"Test acc: {best_test_acc}")
print()
print("Val Accs:", best_val_acc)
print("Test Accs:", best_test_acc)
print(f"Number of params: {count_parameters(args)}")


# %% Generate plots of accuracy and loss vs epochs

fig = plt.figure(figsize=(15, 12))
ax = fig.gca()
ax.tick_params(labelright=True)
for y, label in zip([train_accs, val_accs, test_accs], ["train acc", "val acc", "test acc"]):
    plt.plot(range(args.n_epochs), y, label=label)
ax.legend(prop={'size': 20})
ax.tick_params(axis='both', labelsize = 20)
plt.title("Accuracy vs Epochs", fontsize=30)
plt.ylabel('Accuracy', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.grid(which="major", color="silver", linestyle="dotted")
plt.grid(which="minor", color="silver", linestyle="dotted")
#plt.tight_layout()
plt.savefig(foldername + "/gat_accuracy.png", bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(15, 12))
ax = fig.gca()
ax.tick_params(labelright=True)
for y, label in zip([train_losses, val_losses, test_losses], 
                    ["train loss", "val loss", "test loss"]):
    plt.plot(range(args.n_epochs), y, label=label)
ax.legend(prop={'size': 20})
ax.tick_params(axis='both', labelsize = 20)
plt.title("Loss vs Epochs", fontsize=30)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.grid(which="major", color="silver", linestyle="dotted")
plt.grid(which="minor", color="silver", linestyle="dotted")
#plt.tight_layout()
plt.savefig(foldername + "/gat_loss.png", bbox_inches='tight')
plt.show()

# %% Generate histogram of predicted labels

category_names = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
                  "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
                  "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
                  "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
                  "cs.SD", "cs.SE", "cs.SI", "cs.SY"]

# Get predicted categories
feat = graph.ndata["feat"]
pred = model(graph, feat)
pred = pred.argmax(dim=-1, keepdim=True)

# Split predicted cateogories by train, validate and test sets
train_pred = th.flatten(pred[train_idx]).numpy()
val_pred = th.flatten(pred[val_idx]).numpy()
test_pred = th.flatten(pred[test_idx]).numpy()

# Get the ground truth labels for train set for sorting order later
train_labels = th.flatten(labels[train_idx]).numpy()

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

# %% Save the data

dict = {'predicted':pred, 'args':args, 'num_params': count_parameters(args),
        'accs':accs, 'train_accs':train_accs, 'val_accs':val_accs,'test_accs':test_accs, 
        'losses':losses, 'train_losses':train_losses, 'val_losses':val_losses, 'test_losses':test_losses}
io.savemat(foldername+"/model_results.mat", dict)

#Info on saving/loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html

#Save model state only to make predictions
th.save(model.state_dict(), foldername + "/model_stateonly.pth")

#Save entire model and optimizer state so we can load and keep training
th.save({
            'epoch': epoch,
            'args':args,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accs':accs, 'train_accs':train_accs, 'val_accs':val_accs,'test_accs':test_accs, 
            'losses':losses, 'train_losses':train_losses, 'val_losses':val_losses, 'test_losses':test_losses
            }, foldername + "/checkpoint.pth")

# %% To load the model we would do:

# #Get args and also unpack everything else
# checkpoint = torch.load(foldername + "/checkpoint.pth")
# args = checkpoint['args']
# starting_epoch = checkpoint['epoch']
# accs = checkpoint['accs']
# train_accs, val_accs, test_accs = checkpoint['train_accs'], checkpoint['val_accs'], checkpoint['test_accs']
# losses = checkpoint['losses']
# train_losses, val_losses, test_losses = checkpoint['train_losses'], checkpoint['val_losses'], checkpoint['test_losses']

# #Re-initialize the model and the optimizer
# model = gen_model(args)
# model = model.to(device)
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

# #Load the states saved in the checkpoint
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
train_labels = th.flatten(labels[train_idx]).numpy()
val_labels = th.flatten(labels[val_idx]).numpy()
test_labels = th.flatten(labels[test_idx]).numpy()

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
plot_mat(train_matrix, xlabel='Cluster Class', ylabel='Actual Class', title='Normalized Confusion Matrix', num_decimals = 2,
         xticklabels = category_names, yticklabels = category_names,
         size=(35,30), fontsize = {'title':35, 'axislabel':25, 'small':15}, pic_fname = pic_fname)