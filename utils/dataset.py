import scipy.io as sio
import numpy as np
import torch
import os
import networkx as nx
from collections import defaultdict
import random
import tqdm


def load_data(file_path, G1_name, G2_name, use_attr, shuffle='off'):
    """
    Load dataset from .mat file.
    :param file_path: path to the dataset file
    :param G1_name: name of the first graph
    :param G2_name: name of the second graph
    :param use_attr: whether to use input node attributes
    :param shuffle: shuffle options (off/imbalanced/balanced)
    :return:
        adj1, adj2: adjacency matrices of graph G1, G2
        x1, x2: input node attributes of graph G1, G2
        gnd: ground truth node alignments (training + test pairs)
        H: ground truth alignment matrix (training only)
    """
    assert os.path.exists(file_path), f"{file_path} does not exist"

    print(f"Loading {file_path}...")
    data = sio.loadmat(file_path)
    adj_mat1, adj_mat2 = data[G1_name].astype(int), data[G2_name].astype(int)
    if use_attr:
        x1, x2 = data[f'{G1_name}_node_feat'].astype(np.float64), data[f'{G2_name}_node_feat'].astype(np.float64)
        if type(x1) is not np.ndarray:
            x1 = x1.toarray()
        if type(x2) is not np.ndarray:
            x2 = x2.toarray()
    else:
        x1, x2 = None, None
    gnd = data['gnd'].astype(np.int64) - 1
    H = data['H'].astype(int)

    if type(adj_mat1) is not np.ndarray:
        adj_mat1 = adj_mat1.toarray()
    if type(adj_mat2) is not np.ndarray:
        adj_mat2 = adj_mat2.toarray()
    if type(H) is not np.ndarray:
        H = H.toarray()

    adj_mat1 = torch.from_numpy(adj_mat1).int()
    adj_mat2 = torch.from_numpy(adj_mat2).int()
    x1 = torch.from_numpy(x1).to(torch.float64) if x1 is not None else None
    x2 = torch.from_numpy(x2).to(torch.float64) if x2 is not None else None
    gnd = torch.from_numpy(gnd).long()
    H = torch.from_numpy(H).int()

    n1 = adj_mat1.shape[0]
    n2 = adj_mat2.shape[0]
    num_train = H.sum()

    edge_index1 = np.array(np.where(adj_mat1 == 1)).T
    num_nodes1 = adj_mat1.shape[0]
    G1 = nx.Graph()
    G1.add_nodes_from(np.arange(num_nodes1))
    G1.add_edges_from(edge_index1)
    for edge in G1.edges():
        G1[edge[0]][edge[1]]['weight'] = 1

    # partitions = nx.community.louvain_communities(G1)
    # partitions = sorted(partitions, key=lambda x: len(x))
    #
    # partition_map = {node: idx for idx, com in enumerate(partitions) for node in com}
    # stat = defaultdict(int)
    # train = torch.where(H.T == 1)[0]
    # for node in train:
    #     stat[partition_map[int(node)]] += 1
    # print(f"Distribution of given anchor nodes: {[(i, stat[i]) for i in range(len(partitions))]}")
    #
    # if shuffle != 'off':
    #     H = community_shuffle(n1, n2, G1, partitions, gnd, int(num_train), shuffle)

    return adj_mat1, adj_mat2, x1, x2, gnd, H


def community_shuffle(n1, n2, G, partitions, gnd, num_train, shuffle):
    """
    Shuffle training anchor set based on community.
    :param n1: number of nodes in graph1
    :param n2: number of nodes in graph2
    :param G: input graph (graph1)
    :param partitions: community partitions
    :param gnd: ground truth node alignments
    :param num_train: number of training anchors
    :param shuffle: shuffle options (imbalanced/balanced)
    :return: training anchor set
    """

    anchor_nodes1 = gnd[:, 0]
    anchor_map = {int(pair[0]): int(pair[1]) for pair in gnd}

    H = torch.zeros((n2, n1))
    stat = defaultdict(int)
    cnt = 0

    if shuffle == 'imbalanced':
        for idx, com in enumerate(partitions):
            for node in com:
                if node in anchor_nodes1:
                    H[anchor_map[node], node] = 1
                    stat[idx] += 1
                    cnt += 1
                    if cnt >= num_train:
                        break
            if cnt >= num_train:
                break

    elif shuffle == 'balanced':
        num_nodes_per_com = [len(com) for com in partitions]
        budgets = []
        for num in num_nodes_per_com:
            budgets.append(int(num * num_train / n1))
        budgets[-1] += num_train - sum(budgets)

        for idx, com in enumerate(partitions):
            partition = list(com)
            random.shuffle(partition)
            for node in partition:
                if node in anchor_nodes1:
                    H[anchor_map[node], node] = 1
                    stat[idx] += 1
                    cnt += 1
                    budgets[idx] -= 1
                    if budgets[idx] <= 0:
                        break
            if cnt >= num_train:
                break

        if cnt < num_train:
            partition_map = {node: idx for idx, com in enumerate(partitions) for node in com}
            while cnt < num_train:
                node = int(random.choice(anchor_nodes1))
                if H[anchor_map[node], node] == 0:
                    H[anchor_map[node], node] = 1
                    stat[partition_map[node]] += 1
                    cnt += 1

    print(f"Distribution of shuffled anchor nodes: {[(i, stat[i]) for i in range(len(partitions))]}")

    return H.int()


def perturb_edges(adj, ratio):
    """
    Adding structural noise through edge perturbation.
    :param adj: adjacency matrix of the graph
    :param ratio: edge noise ratio
    :return:
    """

    num_nodes, num_edges = adj.shape[0], int(adj.sum() / 2)
    num_perturb_edges = int(num_edges * ratio)

    with tqdm.tqdm(total=num_perturb_edges, desc="Adding edge noise") as pbar:
        cnt = 0
        while cnt < num_perturb_edges:
            u, v = np.random.randint(0, num_nodes), np.random.randint(0, num_nodes)
            if adj[u, v] == 1:
                adj[u, v] = 0
                adj[v, u] = 0
                if not nx.is_connected(nx.from_numpy_array(adj.numpy())):
                    adj[u, v] = 1
                    adj[v, u] = 1
                else:
                    cnt += 1
                    pbar.update(1)
            else:
                adj[u, v] = 1
                adj[v, u] = 1
                cnt += 1
                pbar.update(1)

    return adj


def perturb_attr(x, ratio, strong_noise=False):
    """
    Adding attribute noise through feature perturbation.
    :param x: input node attributes
    :param ratio: noise ratio
    :param strong_noise: whether to use strong noise
    :return: perturbed node attributes
    """

    num_node, num_attr = x.shape
    num_perturb_attrs = int(num_attr * ratio)

    if strong_noise:
        for idx in range(num_node):
            perturbed_attr = np.random.choice(num_attr, num_perturb_attrs, replace=False)
            x[:, perturbed_attr] = 1 - x[:, perturbed_attr]

    else:
        perturbed_attr = np.random.choice(num_attr, num_perturb_attrs, replace=False)
        x[:, perturbed_attr] = 1 - x[:, perturbed_attr]

    return x
