# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:58:06 2021

@author: glifi
"""
import math
import time
import numpy as np

from scipy import io
from sklearn import metrics
import itertools
from matplotlib import pyplot as plt

import torch as th
from torch import nn
import torch.optim as optim
import torch.utils.data as td
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


foldername = 'logistic regression'

# %% Load dataset
dataset = DglNodePropPredDataset(name = 'ogbn-arxiv')

#Get training, validation and test set indicies
split_idx = dataset.get_idx_split()
train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

#Get the graph, labels, and feature vectors
graph, labels = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
n_classes = dataset.num_classes #num_classes should be 40
features = graph.ndata['feat'] #feature vectors

#Load evaluator
evaluator = Evaluator(name="ogbn-arxiv")

# Define Cross-Entropy-Loss Criterion
cross_entropy = nn.CrossEntropyLoss()

# Define accuracy function
def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]

#Define evaluation function
@th.no_grad()
def evaluate(model, feats, labels, train_idx, val_idx, test_idx, evaluator):

    pred = model(feats)
    train_loss = cross_entropy(pred[train_idx], th.flatten(labels[train_idx]))
    val_loss = cross_entropy(pred[val_idx], th.flatten(labels[val_idx]))
    test_loss = cross_entropy(pred[test_idx], th.flatten(labels[test_idx]))

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )

def count_parameters(model):
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

#Put the train data in pytorch Dataset format
class Dataset(td.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, features, labels):
        'Initialization'
        self.labels = labels
        self.features = features
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.features[ID]
        y = self.labels[ID]

        return X, y

#Get batched data loader for training set
batch_size = 64
train_Dataset = Dataset(train_idx, features, labels)
train_loader = td.DataLoader(train_Dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True)

# %% Define logistic regression model

in_dim = 128 #feature vector size for each paper
out_dim = 40 #number of paper categories

#Define model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

# Initialize model and define optimizer
model = LogisticRegression(in_dim, out_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# %% Train the model

num_epochs = 100
log_every = 5

#save accuracy and loss values
train_accs, val_accs, test_accs = [], [], []
train_losses, val_losses, test_losses = [], [], []

total_time = 0
best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

for epoch in range(1, num_epochs+1):
    # #Keep track of total loss for the epoch
    # total_loss = 0 #total loss
    
    #Time each epoch
    print("Starting Epoch ", epoch)
    tic = time.time()
    
    for i, (feat, label) in enumerate(train_loader):
    
        #Calculate model prediction and loss
        pred = model(feat)
        loss = cross_entropy(pred, th.flatten(label))
        
        # #Add to the average training loss
        # total_loss = total_loss + loss.item()
        
        #Run the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    toc = time.time()
    total_time += toc - tic
    print("Epoch run-time: ", toc - tic)

    #Evalueate the model and save accuracy and loss after each epoch
    train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
        model, features, labels, train_idx, val_idx, test_idx, evaluator
        )
    
    for l, e in zip(
        [train_accs, val_accs, test_accs, train_losses, val_losses, test_losses],
        [train_acc, val_acc, test_acc, train_loss, val_loss, test_loss],
    ):
        l.append(e)
        
    #Update the best validation and test loss/accuracy
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_test_acc = test_acc

    ## Print results for the epoch
    if epoch % log_every == 0:
        print(f"\nEpoch: {epoch}/{num_epochs}")
        print(
            f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
            f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}\n"
        )
    
# %% Printouts

print("*" * 50)
print(f"Average epoch time: {total_time / num_epochs}")
print(f"Total Time: {total_time}")
print(f"Test acc: {best_test_acc}")
print()
print("Val Accs:", best_val_acc)
print("Test Accs:", best_test_acc)
print(f"Number of params: {count_parameters(model)}")
    
# %% Generate plots of accuracy and loss vs epochs

fig = plt.figure(figsize=(15, 12))
ax = fig.gca()
ax.tick_params(labelright=True)
for y, label in zip([train_accs, val_accs, test_accs], ["train acc", "val acc", "test acc"]):
    plt.plot(range(num_epochs), y, label=label)
ax.legend(prop={'size': 20})
ax.tick_params(axis='both', labelsize = 20)
plt.title("Accuracy vs Epochs", fontsize=30)
plt.ylabel('Accuracy', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.grid(which="major", color="silver", linestyle="dotted")
plt.grid(which="minor", color="silver", linestyle="dotted")
#plt.tight_layout()
plt.savefig(foldername + "/accuracy.png", bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(15, 12))
ax = fig.gca()
ax.tick_params(labelright=True)
for y, label in zip([train_losses, val_losses, test_losses], 
                    ["train loss", "val loss", "test loss"]):
    plt.plot(range(num_epochs), y, label=label)
ax.legend(prop={'size': 20})
ax.tick_params(axis='both', labelsize = 20)
plt.title("Loss vs Epochs", fontsize=30)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.grid(which="major", color="silver", linestyle="dotted")
plt.grid(which="minor", color="silver", linestyle="dotted")
#plt.tight_layout()
plt.savefig(foldername + "/loss.png", bbox_inches='tight')
plt.show()

# %% Generate histogram of predicted labels

category_names = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
                  "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
                  "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
                  "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
                  "cs.SD", "cs.SE", "cs.SI", "cs.SY"]

# Get predicted categories
features = graph.ndata["feat"]
pred = model(features)
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

dict = {'predicted':pred, 'num_params': count_parameters(model),
        'train_accs':train_accs, 'val_accs':val_accs,'test_accs':test_accs, 
        'train_losses':train_losses, 'val_losses':val_losses, 'test_losses':test_losses}
io.savemat(foldername+"/model_results.mat", dict)

#Info on saving/loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html

#Save model state only to make predictions
th.save(model.state_dict(), foldername + "/model_stateonly.pth")

#Save entire model and optimizer state so we can load and keep training
th.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accs':train_accs, 'val_accs':val_accs,'test_accs':test_accs, 
            'train_losses':train_losses, 'val_losses':val_losses, 'test_losses':test_losses
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
plot_mat(test_matrix, xlabel='Cluster Class', ylabel='Actual Class', title='Normalized Confusion Matrix', num_decimals = 2,
         xticklabels = category_names, yticklabels = category_names,
         size=(35,30), fontsize = {'title':35, 'axislabel':25, 'small':15}, pic_fname = pic_fname)