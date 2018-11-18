import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from scipy import stats

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    idx=np.array(idx)
    mask[idx-1] = 1
    return np.array(mask, dtype=np.bool)

def load_graph_data(dataset_str):
    import json
    fp=open("number_cosponsor.json","r")
    info = fp.read()
    member_cosponsor = json.loads(info)
    member_list=[]
    for item in member_cosponsor:
        if item[0] not in member_list:
            member_list.append(item[0])
        if item[1] not in member_list:
            member_list.append(item[1])
    member_idx=np.sort(member_list)
    graph={}
    for item in member_cosponsor:
        if item[0] not in graph:
            graph[item[0]]=[item[1]]
        else:
            graph[item[0]].append(item[1])
    DG=nx.MultiDiGraph()
    for item in graph:
        for ITEM in graph[item]:
            DG.add_edge(item,ITEM,weight=1)
    weighted_adj=nx.to_numpy_matrix(DG)
    return member_idx,weighted_adj
def load_vote_data(dataset_str):
    import json
    fp=open("number_vote.json","r")
    info = fp.read()
    member_vote= json.loads(info)
    vote_list=[]
    description_list=[]
    for item in member_vote:
        if item not in vote_list:
            vote_list.append(int(item))
        for word in member_vote[item][0]:
            if word not in description_list:
                description_list.append(int(word))
    vote_list = np.sort(vote_list)
    description_list=np.sort(description_list)
    return member_vote,vote_list,description_list
def vote_vector(vote_number,member_vote,member_idx):
    member_dim=len(member_idx)
    select_vote=member_vote[vote_number]
    y=np.zeros((member_dim,3))
    description=select_vote[0]
    vote_member=[]
    for item in select_vote[1]:
        member=item[0]-1
        vote_member.append(item[0])
        if item[1]==1:
            y[member][0]=1
        if item[1]==0:
            y[member][1]=1
        if item[1]==-1:
            y[member][2]=1
    y_mask=sample_mask(vote_member,member_dim)
    return description,y,y_mask
def padding(list,padnumber):
    if len(list)>=padnumber:
        list=list[-padnumber:]
        list=np.array(list, dtype=np.int32)
    if len(list)< padnumber:
        pad=np.zeros(padnumber-len(list))
        list=np.array(list, dtype=np.int32)
        list=np.append(pad,list)
    return list


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def threshhold(adj,holdnumber):
    adj = stats.threshold(adj, threshmin=holdnumber, newval=0)
    adj=stats.threshold(adj,threshmax=holdnumber,newval=1)

    return adj

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    #adj_normalized = normalize_adj(normalize_adj(adj) + sp.eye(adj.shape[0]))
    #print('non adj')
    adj_normalized=sp.eye(adj.shape[0])
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(description, support, labels, labels_mask, placeholders,tf_idf,vote_number):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['description']: description})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['tfidf_feature']:tf_idf})
    feed_dict.update({placeholders['vote_number']: vote_number})
    #feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict



def idf_construct_feed_dict(description, support, labels, labels_mask, placeholders,tf_idf,vote_number):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['description']: description})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['tfidf_feature']:tf_idf})
    feed_dict.update({placeholders['vote_number']: vote_number})
    #feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict
def multi_task_construct_feed_dic(description, support, labels, labels_mask, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['description']: description})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})

    vote_rate = np.sum(labels, axis=0)
    vote_rate/=np.sum(vote_rate)
    vote_rate=np.reshape(vote_rate,(1,3))
    feed_dict.update({placeholders['vote_rate']:vote_rate})
    return feed_dict

def valtest(y,y_mask,vote_list):
    val_vote = np.random.randint(1, high=len(vote_list) + 1, size=(1, 30))[0]
    y_val_mask = np.zeros(y.shape[0])
    y_train_mask = np.zeros(y.shape[0])
    for item in range(len(y_mask)):
        if y_mask[item] ==True:
            if item in val_vote:
                y_val_mask[item]=1
                y_train_mask[item]=0
            else:
                y_train_mask[item] = 1
                y_val_mask[item] = 0
    y_train_mask=np.array(y_train_mask, dtype=np.bool)
    y_val_mask=np.array(y_val_mask,dtype=np.bool)

    return y_train_mask, y_val_mask
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
