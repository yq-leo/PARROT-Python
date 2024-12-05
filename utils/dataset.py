import scipy.io as sio
import numpy as np
import torch
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

    print(f"Loading {file_path}...")
    data = sio.loadmat(file_path)
    adj_mat1, adj_mat2 = data[G1_name].astype(int), data[G2_name].astype(int)
    if use_attr:
        x1, x2 = data[f'{G1_name}_node_feat'].astype(np.float64), data[f'{G2_name}_node_feat'].astype(np.float64)
        if type(x1) is not np.ndarray:
            x1 = x1.A
        if type(x2) is not np.ndarray:
            x2 = x2.A
    else:
        x1, x2 = None, None
    gnd = data['gnd'].astype(np.int64) - 1
    H = data['H'].astype(int)

    if type(adj_mat1) is not np.ndarray:
        adj_mat1 = adj_mat1.A
    if type(adj_mat2) is not np.ndarray:
        adj_mat2 = adj_mat2.A
    if type(H) is not np.ndarray:
        H = H.A

    adj_mat1 = torch.from_numpy(adj_mat1).to(torch.int8)
    adj_mat2 = torch.from_numpy(adj_mat2).to(torch.int8)
    x1 = torch.from_numpy(x1).to(torch.float64) if x1 is not None else None
    x2 = torch.from_numpy(x2).to(torch.float64) if x2 is not None else None
    gnd = torch.from_numpy(gnd)
    H = torch.from_numpy(H).to(torch.int8)

    return adj_mat1, adj_mat2, x1, x2, gnd, H
