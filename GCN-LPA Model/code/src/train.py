import numpy as np
import tensorflow as tf
from auxiliary import get_save_paths
from model import GCN_LPA
import os.path
from os import mkdir

from auxiliary import get_save_paths, load_training_vars

def print_statistics(features, labels, adj):
    n_nodes = features[2][0]
    n_edges = (len(adj[0]) - labels.shape[0]) // 2
    n_features = features[2][1]
    n_labels = labels.shape[1]
    labeled_node_rate = 20 * n_labels / n_nodes

    n_intra_class_edge = 0
    for i, j in adj[0]:
        if i < j and np.argmax(labels[i]) == np.argmax(labels[j]):
            n_intra_class_edge += 1
    intra_class_edge_rate = n_intra_class_edge / n_edges

    print('n_nodes: %d' % n_nodes)
    print('n_edges: %d' % n_edges)
    print('n_features: %d' % n_features)
    print('n_labels: %d' % n_labels)
    print('labeled node rate: %.4f' % labeled_node_rate)
    print('intra-class edge rate: %.4f' % intra_class_edge_rate)

default_training_vars = {
    'num_epochs': 0,
    'train_accs': [],
    'val_accs': [],
    'test_accs': [],
    'train_losses': [],
    'val_losses': [],
    'test_losses': [],
}

def train(args, data, batch_test=False, save_dir=None):
    '''
    Trains the GCN-LPA model on the given data.
    Parameters:
    save_dir -- if not None, attempts to load a previous training state,
                and also saves at the end
    '''
    
    print("Creating Model")

    features, labels, adj, train_mask, val_mask, test_mask = [data[i] for i in range(6)]

    # uncomment the next line if you want to print statistics of the current dataset
    #print_statistics(features, labels, adj)
    model = GCN_LPA(args, features, labels, adj)

    def get_feed_dict(mask, dropout):
        feed_dict = {model.label_mask: mask, model.dropout: dropout}
        return feed_dict

    print("Preparing to train model")
    
    if save_dir:
        if not os.path.isdir(save_dir):
            mkdir(save_dir)
        # The paths for the save/loads
        save_path_model, save_path_training_vars = get_save_paths(save_dir)
        saver = tf.train.Saver()

    successful_load = True

    # Attempt to load the training vars
    if save_dir and os.path.isfile(save_path_training_vars):
        training_vars = load_training_vars(save_path_training_vars)
        # Check that it's what we expect (at least somewhat)
        assert default_training_vars.keys() == training_vars.keys()
        print("Successfully Loaded training_vars")
    else:
        successful_load = False
    
    with tf.Session() as sess:

        # Attempt to load the model
        #TODO: Replace with more robust version
        if save_dir and os.path.isfile(save_path_model + ".meta"):
            # Load the model state from a file
            saver.restore(sess, save_path_model)
            print("Successfully Loaded model")
        else:
            successful_load = False
        
        if not successful_load:
            training_vars = default_training_vars
            # Only run initializer when not loaded
            sess.run(tf.global_variables_initializer())

        print("Training Model")

        best_val_acc = 0
        final_test_acc = 0
        for epoch in range(args.epochs):

            # train
            _, train_loss, train_acc = sess.run(
                [model.optimizer, model.loss, model.accuracy], feed_dict=get_feed_dict(train_mask, args.dropout))
            training_vars['train_accs'].append(train_acc)
            training_vars['train_losses'].append(train_loss)
            
            # validation
            val_loss, val_acc = sess.run([model.loss, model.accuracy], feed_dict=get_feed_dict(val_mask, 0.0))
            training_vars['val_accs'].append(val_acc)
            training_vars['val_losses'].append(val_loss)

            # test
            test_loss, test_acc = sess.run([model.loss, model.accuracy], feed_dict=get_feed_dict(test_mask, 0.0))
            training_vars['test_accs'].append(test_acc)
            training_vars['test_losses'].append(test_loss)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

            training_vars['num_epochs'] += 1

            if not batch_test:
                print('epoch %d   train loss: %.4f  acc: %.4f   val loss: %.4f  acc: %.4f   test loss: %.4f  acc: %.4f'
                      % (training_vars['num_epochs'], train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

        if not batch_test:
            print('final test acc: %.4f' % final_test_acc)
        else:
            return final_test_acc

        if save_dir:
            # Save the model
            path = saver.save(sess, save_path_model)
            print("Model saved in path: %s" % path)
            # Save the training variables
            np.save(save_path_training_vars, training_vars)

    return training_vars, model