'''
I used these to work through bugs and identify data's format
'''

def is_symmetric_sparse_arr(arr):
    '''
    arr is the coord array of the sparse array tuple
    '''
    for row in arr:
        a,b = row[0], row[1]
        if not [b,a] in arr:
            return False
    else:
        return True

def has_diagonal(sparse_tuple):
    indices = sparse_tuple[0]
    values = sparse_tuple[1]
    shape = sparse_tuple[2]
    N = shape[0]
    ans_1 = sum([r[0] == r[1] for r in indices])
    return ans_1 == N

def only_ones(sparse_tuple):
    indices = sparse_tuple[0]
    values = sparse_tuple[1]
    shape = sparse_tuple[2]
    largest = np.max(values)
    smallest = np.min(values)
    return largest == smallest and largest == 1

def get_save_paths(save_dir):
    save_path_model = save_dir + "model.ckpt"
    save_path_training_vars = save_dir + "vars.npy"
    return save_path_model, save_path_training_vars

from numpy.lib.npyio import save
from tensorflow.python.framework.c_api_util import tf_output
from model import GCN_LPA

import numpy as np

def load_training_vars(save_path_training_vars):
    # Load the training variables from a file
    training_vars = np.load(save_path_training_vars, allow_pickle = True)
    # For some reason, it's wrapped in an array
    training_vars = training_vars[()]
    return training_vars

def load_train_result(args, data, save_dir):
    # Reset tensorflow to default
    #tf.reset_default_graph()
    features, labels, adj, train_mask, val_mask, test_mask = [data[i] for i in range(6)]
    model = GCN_LPA(args, features, labels, adj)
    _, save_path_training_vars = get_save_paths(save_dir)
    training_vars = load_training_vars(save_path_training_vars)

    return training_vars, model

import tensorflow as tf

def evaluate_model(data, model, save_dir):
    '''
    The GCNLPA only.
    '''
    features, labels, adj, train_mask, val_mask, test_mask = [data[i] for i in range(6)]

    # Recover the model outputs
    saver = tf.train.Saver()
    with tf.Session() as sess:

        # Restore variables from disk.
        save_path_model, _ = get_save_paths(save_dir)
        saver.restore(sess, save_path_model)

        # For some reason, the way the model is built, you need to give a mask
        # The return values is the same regardless of model.label_mask
        output = sess.run(model.outputs, feed_dict={model.label_mask: test_mask, model.dropout: 0})

    return output