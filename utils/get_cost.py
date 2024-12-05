import torch
from torch.nn.functional import softmax
import numpy as np
import time
import os
from tqdm import tqdm


def get_cost(dataset, A1, A2, X1, X2, H, rwrIter, rwIter, alpha, beta, gamma, rwr_time_list):
    """
    Calculate cross/intra-graph cost based on attribute/rw
    :param dataset: dataset name
    :param A1: adjacency matrix of graph G1, shape=(n1, n1)
    :param A2: adjacency matrix of graph G2, shape=(n2, n2)
    :param X1: input node attributes of graph G1, shape=(n1, d)
    :param X2: input node attributes of graph G2, shape=(n2, d)
    :param H: anchor links, shape=(n2, n1)
    :param rwrIter: number of iteartion for rwr on separated graphs
    :param rwIter: number of iteration for rwr on product graph
    :param alpha: weight balancing attribute cost and rwr cost
    :param beta: rwr restart ratio
    :param gamma: discounted factor of Bellman equation
    :param rwr_time_list: list to store RWR time
    :return:
        crossC: cross-graph cost matrix, shape=(n1, n2)
        intraC1: intra-graph cost matrix for graph 1, shape=(n1, n1)
        intraC2: intra-graph cost matrix for graph 2, shape=(n2, n2)
    """

    start_time = time.time()

    # calculate RWR
    T1 = cal_trans(A1, None)
    T2 = cal_trans(A2, None)
    start_rwr = time.time()
    rwr1, rwr2 = get_sep_rwr(T1, T2, H, beta, rwrIter)
    rwr_time_list.append(time.time() - start_rwr)
    print(f"RWR time: {rwr_time_list[-1]:.4f}")
    rwrCost = get_cross_cost(rwr1, rwr2, H)

    # cross/intra-graph cost based on node attributes
    if X1 is None or X2 is None:
        X1 = rwr1
        X2 = rwr2

    intraC1 = get_intra_cost(X1) * A1
    intraC2 = get_intra_cost(X2) * A2
    crossC = get_cross_cost(X1, X2, H)

    # rwr on the product graph
    crossC = crossC + alpha * rwrCost
    L1 = A1 / A1.sum(1, keepdim=True).to(torch.float64)
    L2 = A2 / A2.sum(1, keepdim=True).to(torch.float64)

    crossC = get_prod_rwr(L1, L2, crossC, H, beta, gamma, rwIter)

    end_time = time.time()
    print(f"Time for cost matrix: {end_time - start_time:.2f}s")

    if not os.path.exists(f"datasets/rwr"):
        os.makedirs(f"datasets/rwr")
    np.savez(f"datasets/rwr/rwr_cost_{dataset}.npz", rwr1=rwr1.numpy(), rwr2=rwr2.numpy(), cross_rwr=rwrCost.numpy())

    return crossC, intraC1, intraC2


def cal_trans(A, X=None):
    """
    Calculate transition probability based on node attributes
    :param A: adjacency matrix of the graph, shape=(n, n)
    :param X: node attributes of the graph, shape=(n, d)
    :return:
        T: transition probability matrix, shape=(n, n)
    """

    n = A.shape[0]

    if X is None:
        X = torch.ones((n, 1)).to(torch.float64)
    X = X / torch.linalg.norm(X, dim=1, ord=2, keepdim=True)
    sim = X @ X.T
    T = sim * A
    for i in range(n):
        T[i, torch.where(T[i] != 0)[0]] = softmax(T[i, torch.where(T[i] != 0)[0]], dim=0)

    return T


def get_sep_rwr(T1, T2, H, beta, sepRwrIter):
    """
    RWR on separated graphs
    :param T1: transition probability matrix of graph 1, shape=(n1, n1)
    :param T2: transition probability matrix of graph 2, shape=(n2, n2)
    :param H: anchor links, shape=(n2, n1)
    :param beta: restart probability
    :param sepRwrIter: maximum number of iterations for RWR
    :return:
        rwr1: RWR score of graph 1, shape=(n1, num of anchor nodes)
        rwr2: RWR score of graph 2, shape=(n2, num of anchor nodes)
    """
    eps = 1e-5

    anchors1, anchors2 = torch.where(H.T == 1)
    n1, n2 = T1.shape[0], T2.shape[0]
    num_anchors = anchors1.shape[0]

    e1 = torch.zeros((n1, num_anchors)).to(torch.float64)
    e2 = torch.zeros((n2, num_anchors)).to(torch.float64)
    e1[(anchors1, torch.arange(num_anchors))] = 1
    e2[(anchors2, torch.arange(num_anchors))] = 1

    r1 = torch.zeros((n1, num_anchors)).to(torch.float64)
    r2 = torch.zeros((n2, num_anchors)).to(torch.float64)

    for i in tqdm(range(sepRwrIter), desc="Computing separate RWR scores"):
        r1_old = torch.clone(r1)
        r2_old = torch.clone(r2)
        r1 = (1 - beta) * T1 @ r1 + beta * e1
        r2 = (1 - beta) * T2 @ r2 + beta * e2
        diff = torch.max(torch.max(torch.abs(r1 - r1_old)), torch.max(torch.abs(r2 - r2_old)))
        if diff < eps:
            break

    return r1, r2


def get_cross_cost(X1, X2, H):
    """
    Calculate cross-graph cost
    :param X1: node attributes of graph 1, shape=(n1, d)
    :param X2: node attributes of graph 2, shape=(n2, d)
    :param H: anchor links, shape=(n2, n1)
    :return:
        crossCost: alignment cost based on node attributes, shape=(n1, n2)
    """

    _, d = X1.shape
    X1_zero_pos = torch.where(X1.abs().sum(1) == 0)
    X2_zero_pos = torch.where(X2.abs().sum(1) == 0)
    if X1_zero_pos[0].shape[0] != 0:
        X1[X1_zero_pos] = torch.ones(d).to(torch.float64)
    if X2_zero_pos[0].shape[0] != 0:
        X2[X2_zero_pos] = torch.ones(d).to(torch.float64)

    X1 = X1 / torch.linalg.norm(X1, dim=1, ord=2, keepdim=True)
    X2 = X2 / torch.linalg.norm(X2, dim=1, ord=2, keepdim=True)

    crossCost = torch.exp(-(X1 @ X2.T))
    crossCost[torch.where(H.T == 1)] = 0

    return crossCost


def get_intra_cost(X):
    """
    Calculate intra-graph cost, i.e., exp(-V @ V^T)
    :param X: node attributes of the graph, shape=(n, d)
    :return:
        intraCost: alignment cost based on node attributes, shape=(n, n)
    """

    _, d = X.shape
    X_zero_pos = torch.where(X.abs().sum(1) == 0)
    if X_zero_pos[0].shape[0] != 0:
        X[X_zero_pos] = torch.ones(d).to(torch.float64)
    X = X / torch.linalg.norm(X, dim=1, ord=2, keepdim=True)
    intraCost = torch.exp(-(X @ X.T))

    return intraCost


def get_prod_rwr(L1, L2, nodeCost, H, beta, gamma, prodRwrIter):
    """
    RWR cost on product graph
    :param L1: Laplacian matrix of graph 1, shape=(n1, n1)
    :param L2: Laplacian matrix of graph 2, shape=(n2, n2)
    :param nodeCost: cross-graph cost matrix, shape=(n1, n2)
    :param H: anchor links, shape=(n2, n1)
    :param beta: rwr restart ratio
    :param gamma: discounted factor
    :param prodRwrIter: maximum number of iterations for Bellman equation
    :return:
        crossCost: random walk cost, shape=(n1, n2)
    """

    eps = 1e-2
    nx, ny = H.T.shape
    HInd = torch.where(H.T == 1)
    crossCost = torch.zeros((nx, ny)).to(torch.float64)
    for i in tqdm(range(prodRwrIter), desc="Computing product RWR scores"):
        rwCost_old = torch.clone(crossCost)
        crossCost = (1 + gamma * beta) * nodeCost + (1 - beta) * gamma * L1 @ crossCost @ L2.T
        crossCost[HInd] = 0
        if torch.max(torch.abs(crossCost - rwCost_old)) < eps:
            break
    crossCost = (1 - gamma) * crossCost
    crossCost[HInd] = 0

    return crossCost
