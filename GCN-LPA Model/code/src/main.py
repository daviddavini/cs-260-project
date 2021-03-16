## Imports and Setup

print("Importing")
# Suppress all the deprecated warnings!
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
import tensorflow as tf
from time import time
from data_loader import load_data, load_npz, load_random, load_ogb, load_ogb_2, load_ogb_3
from train import train

import os.path
from os import mkdir

import torch

from auxiliary import evaluate_model, load_train_result

from analysis import get_split_pred, plot_accs, plot_losses
from analysis import plot_pred_histograms, plot_contingency_matrices

# Remove TF warnings
tf.logging.set_verbosity(tf.logging.ERROR)

seed = 234
np.random.seed(seed)
tf.set_random_seed(seed)

class default_args:
    dataset = 'ogbn-arxiv'
    epochs = 50 #200
    dim = 16
    gcn_layer = 2
    lpa_iter = 5
    l2_weight = 5e-4
    lpa_weight = 1
    dropout = 0
    lr = 0.2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',    type=str, default=default_args.dataset, help='which dataset to use')
parser.add_argument('--epochs',     type=int, default=default_args.epochs, help='the number of epochs')
parser.add_argument('--dim',        type=int, default=default_args.dim, help='dimension of hidden layers')
parser.add_argument('--gcn_layer',  type=int, default=default_args.gcn_layer, help='number of GCN layers')
parser.add_argument('--lpa_iter',   type=int, default=default_args.lpa_iter, help='number of LPA iterations')
parser.add_argument('--l2_weight',  type=float, default=default_args.l2_weight, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=default_args.lpa_weight, help='weight of LP regularization')
parser.add_argument('--dropout',    type=float, default=default_args.dropout, help='dropout rate')
parser.add_argument('--lr',         type=float, default=default_args.lr, help='learning rate')
args = parser.parse_args()

t = time()

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    data = load_data(args.dataset)
elif args.dataset in ['coauthor-cs', 'coauthor-phy']:
    data = load_npz(args.dataset)
elif args.dataset == 'ogbn-arxiv':
    data = load_ogb_3(args.dataset)
else:
    n_nodes = 1000
    data = load_random(n_nodes=n_nodes, n_train=100, n_val=200, p=10/n_nodes)

def get_foldername(args):
    foldername = "GCN-LPA"
    vars_args = vars(args)
    vars_default_args = vars(default_args)
    for key in vars_args.keys():
        if key == 'epochs':
            continue
        if vars_args[key] != vars_default_args[key]:
            foldername += "_" + key + "={}".format(vars_args[key])
    return foldername

foldername = get_foldername(args)
save_dir = foldername + "/save_1/"

# Create the folder if it doesn't exist
if not os.path.isdir(foldername + "/"):
    mkdir(foldername + "/")

# We either train a model or load a pre-trained model, but not both
do_training = True
if do_training:
    ## Train Model
    training_vars, model = train(args, data, save_dir=save_dir)
else:
    ## Load Model 
    training_vars, model = load_train_result(args, data, save_dir=save_dir)

## Evaluate the model (on all the data)

print("Evaluating the Model")
output = evaluate_model(data, model, save_dir)


## Prepare for General Analysis

print("Preparing for Analysis")

def count_parameters(model):
    param_counts = [np.prod(v.shape.as_list()) for v in model.vars]
    print(param_counts)
    total = sum(param_counts)
    return total

param_count = count_parameters(model)

# Reformat and split the prediction matrix
pred = torch.from_numpy(output)
pred = pred.argmax(dim=-1, keepdim=True)
preds = get_split_pred(pred)

# Unpack training_vars
num_epochs = training_vars['num_epochs']
train_accs = training_vars['train_accs']
val_accs = training_vars['val_accs']
test_accs = training_vars['test_accs']
train_losses = training_vars['train_losses']
val_losses = training_vars['val_losses']
test_losses = training_vars['test_losses']
accs = [train_accs, val_accs, test_accs]
losses = [train_losses, val_losses, test_losses]

# Create Analysis Work on Model
print("Analyzing")
print("Making Plots")
# Generate plots of accuracy and loss vs epochs
plot_accs(foldername, num_epochs, accs)
plot_losses(foldername, num_epochs, losses)

print("Making Histograms")
# Generate histogram of predicted labels
plot_pred_histograms(foldername, preds)

# Save the data
# save_data(foldername, pred, accs, losses, param_count)

print("Making Matrices")
# Plot the contigency matrices
plot_contingency_matrices(foldername, preds)


print('time used: %d s' % (time() - t))