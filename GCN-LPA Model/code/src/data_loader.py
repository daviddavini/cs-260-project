import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    return indices, values, shape


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.float64)


def split_dataset(n_samples):
    val_indices = np.random.choice(list(range(n_samples)), size=int(n_samples * 0.2), replace=False)
    left = set(range(n_samples)) - set(val_indices)
    test_indices = np.random.choice(list(left), size=int(n_samples * 0.2), replace=False)
    train_indices = list(left - set(test_indices))

    train_mask = get_mask(train_indices, n_samples)
    eval_mask = get_mask(val_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)

    return train_mask, eval_mask, test_mask


def load_data(dataset):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open('../data/{}/ind.{}.{}'.format(dataset, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/{}/ind.{}.test.index".format(dataset, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil().astype(np.float64)
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)
    features = sparse_to_tuple(features)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    graph = nx.from_dict_of_lists(graph)
    graph.add_edges_from([(i, i) for i in range(len(graph.nodes)) if not graph.has_edge(i, i)])  # add self-loops
    adj = nx.adjacency_matrix(graph)
    adj = sparse_to_tuple(adj)

    train_mask, val_mask, test_mask = split_dataset(len(graph.nodes))

    print(features[0].shape, features[1].shape, labels.shape, adj[0].shape, adj[1].shape, adj[2][0], adj[2][1], train_mask.shape, val_mask.shape, test_mask.shape)

    return features, labels, adj, train_mask, val_mask, test_mask


def load_npz(dataset):
    file_map = {'coauthor-cs': 'ms_academic_cs.npz', 'coauthor-phy': 'ms_academic_phy.npz'}
    file_name = file_map[dataset]

    with np.load('../data/' + file_name) as f:
        f = dict(f)

        features = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']), shape=f['attr_shape'])
        features = features.astype(np.float64)
        features = normalize_features(features)
        features = sparse_to_tuple(features)

        labels = f['labels'].reshape(-1, 1)
        labels = OneHotEncoder(sparse=False).fit_transform(labels)

        adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']), shape=f['adj_shape'])
        adj += sp.eye(adj.shape[0])  # add self-loops
        adj = sparse_to_tuple(adj)

    train_mask, val_mask, test_mask = split_dataset(labels.shape[0])

    return features, labels, adj, train_mask, val_mask, test_mask


def load_random(n_nodes, n_train, n_val, p):
    features = sp.eye(n_nodes).tocsr()
    features = sparse_to_tuple(features)
    labels = np.ones([n_nodes, 1])
    graph = nx.generators.fast_gnp_random_graph(n_nodes, p=p)
    adj = nx.adjacency_matrix(graph)
    adj = sparse_to_tuple(adj)

    train_mask = np.array([1] * n_train + [0] * (n_nodes - n_train)).astype(np.float64)
    val_mask = np.array([0] * n_train + [1] * n_val + [0] * (n_nodes - n_train - n_val)).astype(np.float64)
    test_mask = np.array([0] * (n_train + n_val) + [1] * (n_nodes - n_train - n_val)).astype(np.float64)
    return features, labels, adj, train_mask, val_mask, test_mask

def indexes2booleanvec(size, indices):
    '''
    Converts a numpy array of indices into a boolean array
    '''
    v = np.zeros(size)
    v[indices] = 1
    return v

def sparse_identity(N):
    '''
    Returns a sparse torch tensor, (symmetric) identity matrix
    '''
    i = [[x,x] for x in range(N)]
    v = [1 for x in range(N)]
    sp = torch.sparse_coo_tensor(list(zip(*i)), v, (N,N))
    return sp

import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

def load_ogb(dataset):
    ## Load the dataset

    ## Setup PyTorch
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    dataset = PygNodePropPredDataset(name=dataset, transform=T.ToSparseTensor())
    ogb_data = dataset[0]
    # TODO: Not sure how to format adj_t...
    ##ogb_data.adj_t = ogb_data.adj_t.to_symmetric()
    ogb_data = ogb_data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    train_idx, valid_idx, split_idx = train_idx.numpy(), valid_idx.numpy(), test_idx.numpy()

    # Convert OGB's data split pytorch index vectors to Wang's data split numpy boolean masks
    train_mask_2 = indexes2booleanvec(ogb_data.num_nodes, train_idx)
    val_mask_2 = indexes2booleanvec(ogb_data.num_nodes, valid_idx)
    test_mask_2 = indexes2booleanvec(ogb_data.num_nodes, test_idx)

    # Convert OGB's adjacency SparseTensor to Wang's adjacency index matrix (Nx2)
    adj_t = ogb_data.adj_t.to_torch_sparse_coo_tensor()
    adj_2_0 = adj_t.coalesce().indices().numpy()
    adj_2_0 = adj_2_0.T.astype('int32')
    ##adj_2_0 = np.vstack((adj_2_0, np.array([[i,i] for i in range(ogb_data.num_nodes)])))
    adj_2_1 = adj_t.coalesce().values().numpy().astype('float64')
    adj_2_2 = tuple(adj_t.size())
    #TODO: Fix the adjacency matrix, bc it probably is symmetric with identity
    adj_2 = (adj_2_0, adj_2_1, adj_2_2)

    from sklearn.preprocessing import OneHotEncoder
    labels_2 = ogb_data.y.numpy()
    labels_2 = OneHotEncoder(sparse=False).fit_transform(labels_2)

    #TODO: I don't know if this feature vector will work
    # OGB used a skip-gram encoding, 
    # whereas Wang's Citeseer just used normalized rows with 1-0 for different words
    x = ogb_data.x + 1.5
    norm_x = np.apply_along_axis(np.linalg.norm, 1, x)
    x = x / norm_x[:,None]
    x = x.to_sparse()
    features_2_0 = x.indices().numpy().T.astype('int32')
    features_2_1 = x.values().numpy()
    #features_2_1 = 1.5 + features_2_1
    features_2_1 = features_2_1.astype('float64')
    features_2_2 = tuple(x.size())
    features_2 = features_2_0, features_2_1, features_2_2

    data2 = features_2, labels_2, adj_2, train_mask_2, val_mask_2, test_mask_2

    return data2

def load_ogb_2(dataset):
    ## Load the dataset

    ## Setup PyTorch
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    dataset = PygNodePropPredDataset(name=dataset, transform=T.ToSparseTensor())
    ogb_data = dataset[0]
    # TODO: Not sure how to format adj_t...
    ogb_data.adj_t = ogb_data.adj_t.to_symmetric()
    ogb_data = ogb_data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    train_idx, valid_idx, split_idx = train_idx.numpy(), valid_idx.numpy(), test_idx.numpy()

    # Convert OGB's data split pytorch index vectors to Wang's data split numpy boolean masks
    train_mask_2 = indexes2booleanvec(ogb_data.num_nodes, train_idx)
    val_mask_2 = indexes2booleanvec(ogb_data.num_nodes, valid_idx)
    test_mask_2 = indexes2booleanvec(ogb_data.num_nodes, test_idx)

    # Add 1's down the diagonal of adj_t
    adj_t = ogb_data.adj_t.to_torch_sparse_coo_tensor()
    adj_t = adj_t + sparse_identity(adj_t.shape[0])
    # Convert OGB's adjacency SparseTensor to Wang's adjacency index matrix (Nx2)
    adj_2_0 = adj_t.coalesce().indices().numpy()
    adj_2_0 = adj_2_0.T.astype('int32')
    ##adj_2_0 = np.vstack((adj_2_0, np.array([[i,i] for i in range(ogb_data.num_nodes)])))
    adj_2_1 = adj_t.coalesce().values().numpy().astype('float64')
    adj_2_2 = tuple(adj_t.size())
    #TODO: Fix the adjacency matrix, bc it probably is symmetric with identity
    adj_2 = (adj_2_0, adj_2_1, adj_2_2)

    from sklearn.preprocessing import OneHotEncoder
    labels_2 = ogb_data.y.numpy()
    labels_2 = OneHotEncoder(sparse=False).fit_transform(labels_2)

    #TODO: I don't know if this feature vector will work
    # OGB used a skip-gram encoding, 
    # whereas Wang's Citeseer just used normalized rows with 1-0 for different words
    x = ogb_data.x + 1.5
    norm_x = np.apply_along_axis(np.linalg.norm, 1, x)
    x = x / norm_x[:,None]
    x = x.to_sparse()
    features_2_0 = x.indices().numpy().T.astype('int32')
    features_2_1 = x.values().numpy()
    #features_2_1 = 1.5 + features_2_1
    features_2_1 = features_2_1.astype('float64')
    features_2_2 = tuple(x.size())
    features_2 = features_2_0, features_2_1, features_2_2

    data2 = features_2, labels_2, adj_2, train_mask_2, val_mask_2, test_mask_2

    return data2

def load_ogb_3(dataset):

    # NOTE: Since torch_geometric takes FOREVER to load, we cache the ogb data
    # to avoid repeating the torch-conversion hassle each time

    # Check if the loaded dataset is cached. If it is, just return it.

    ## Load the dataset

    print("Loading OGB dataset")

    ## Setup PyTorch
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    dataset = PygNodePropPredDataset(name=dataset, transform=T.ToSparseTensor())
    ogb_data = dataset[0]
    # TODO: Not sure how to format adj_t...
    ogb_data.adj_t = ogb_data.adj_t.to_symmetric()
    ogb_data = ogb_data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    train_idx, valid_idx, split_idx = train_idx.numpy(), valid_idx.numpy(), test_idx.numpy()

    # Convert OGB's data split pytorch index vectors to Wang's data split numpy boolean masks
    train_mask_2 = indexes2booleanvec(ogb_data.num_nodes, train_idx)
    val_mask_2 = indexes2booleanvec(ogb_data.num_nodes, valid_idx)
    test_mask_2 = indexes2booleanvec(ogb_data.num_nodes, test_idx)

    # Add 1's down the diagonal of adj_t
    adj_t = ogb_data.adj_t.to_torch_sparse_coo_tensor()
    adj_t = adj_t + sparse_identity(adj_t.shape[0])
    # Convert OGB's adjacency SparseTensor to Wang's adjacency index matrix (Nx2)
    adj_2_0 = adj_t.coalesce().indices().numpy()
    adj_2_0 = adj_2_0.T.astype('int32')
    ##adj_2_0 = np.vstack((adj_2_0, np.array([[i,i] for i in range(ogb_data.num_nodes)])))
    adj_2_1 = adj_t.coalesce().values().numpy().astype('float64')
    adj_2_2 = tuple(adj_t.size())
    #TODO: Fix the adjacency matrix, bc it probably is symmetric with identity
    adj_2 = (adj_2_0, adj_2_1, adj_2_2)

    from sklearn.preprocessing import OneHotEncoder
    labels_2 = ogb_data.y.numpy()
    labels_2 = OneHotEncoder(sparse=False).fit_transform(labels_2)

    #TODO: I don't know if this feature vector will work
    # OGB used a skip-gram encoding, 
    # whereas Wang's Citeseer just used normalized rows with 1-0 for different words
    x = ogb_data.x
    x = x.to_sparse()
    features_2_0 = x.indices().numpy().T.astype('int32')
    features_2_1 = x.values().numpy().astype('float64')
    features_2_2 = tuple(x.size())
    features_2 = features_2_0, features_2_1, features_2_2

    data2 = features_2, labels_2, adj_2, train_mask_2, val_mask_2, test_mask_2

    #import pdb; pdb.set_trace()

    return data2