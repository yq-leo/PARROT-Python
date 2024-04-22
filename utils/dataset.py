import scipy.io as sio
import numpy as np
import os


def load_data(file_path, G1_name, G2_name, use_attr):
    """
    Load dataset from .mat file.
    :param file_path: path to the dataset file
    :param G1_name: name of the first graph
    :param G2_name: name of the second graph
    :param use_attr: whether to use input node attributes
    :return:
        adj1, adj2: adjacency matrices of graph G1, G2
        x1, x2: input node attributes of graph G1, G2
        gnd: ground truth node alignments (training + test pairs)
        H: ground truth alignment matrix (training only)
    """
    assert os.path.exists(file_path), f"{file_path} does not exist"

    data = sio.loadmat(file_path)
    adj_mat1, adj_mat2 = data[G1_name].astype(int), data[G2_name].astype(int)
    if use_attr:
        x1, x2 = data[f'{G1_name}_node_feat'].astype(np.float64), data[f'{G2_name}_node_feat'].astype(np.float64)
    else:
        x1, x2 = None, None
    gnd = data['gnd'].astype(np.int64) - 1
    H = data['H'].astype(int)

    if type(adj_mat1) is not np.ndarray:
        adj_mat1 = adj_mat1.A
    if type(adj_mat2) is not np.ndarray:
        adj_mat2 = adj_mat2.A

    return adj_mat1, adj_mat2, x1, x2, gnd, H
