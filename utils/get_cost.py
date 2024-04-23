import numpy as np
from scipy.special import softmax
import time
from tqdm import tqdm


def get_cost(A1, A2, X1, X2, H, rwrIter, rwIter, alpha, beta, gamma):
    """
    Calculate cross/intra-graph cost based on attribute/rw
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
    :return:
        crossC: cross-graph cost matrix, shape=(n1, n2)
        intraC1: intra-graph cost matrix for graph 1, shape=(n1, n1)
        intraC2: intra-graph cost matrix for graph 2, shape=(n2, n2)
    """

    start_time = time.time()

    # calculate RWR
    T1 = cal_trans(A1, None)
    T2 = cal_trans(A2, None)
    rwr1, rwr2 = get_sep_rwr(T1, T2, H, beta, rwrIter)
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
    L1 = A1 / A1.sum(1, keepdims=True)
    L2 = A2 / A2.sum(1, keepdims=True)

    crossC = get_prod_rwr(L1, L2, crossC, H, beta, gamma, rwIter)

    end_time = time.time()
    print(f"Time for cost matrix: {end_time - start_time:.2f}s")

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
        X = np.ones((n, 1)).astype(np.float64)
    X = X / np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    sim = X @ X.T
    T = sim * A
    for i in range(n):
        T[i, np.where(T[i] != 0)] = softmax(T[i, np.where(T[i] != 0)])

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

    anchors1, anchors2 = np.where(H.T == 1)
    n1, n2 = T1.shape[0], T2.shape[0]
    num_anchors = anchors1.shape[0]

    e1 = np.zeros((n1, num_anchors)).astype(np.float64)
    e2 = np.zeros((n2, num_anchors)).astype(np.float64)
    e1[(anchors1, np.arange(num_anchors))] = 1
    e2[(anchors2, np.arange(num_anchors))] = 1

    r1 = np.zeros((n1, num_anchors)).astype(np.float64)
    r2 = np.zeros((n2, num_anchors)).astype(np.float64)

    for i in tqdm(range(sepRwrIter), desc="Computing separate RWR scores"):
        r1_old = r1.copy()
        r2_old = r2.copy()
        r1 = (1 - beta) * T1 @ r1 + beta * e1
        r2 = (1 - beta) * T2 @ r2 + beta * e2
        diff = max(np.max(np.abs(r1 - r1_old)), np.max(np.abs(r2 - r2_old)))
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
    X1_zero_pos = np.where(np.abs(X1).sum(1) == 0)
    X2_zero_pos = np.where(np.abs(X2).sum(1) == 0)

    X1 = X1 / np.linalg.norm(X1, axis=1, ord=2, keepdims=True)
    X2 = X2 / np.linalg.norm(X2, axis=1, ord=2, keepdims=True)

    if X1_zero_pos[0].shape[0] != 0:
        X1[X1_zero_pos] = np.sqrt(1/d)
    if X2_zero_pos[0].shape[0] != 0:
        X2[X2_zero_pos] = np.sqrt(1/d)
    crossCost = np.exp(-(X1 @ X2.T))
    crossCost[np.where(H.T == 1)] = 0

    return crossCost


def get_intra_cost(X):
    """
    Calculate intra-graph cost, i.e., exp(-V @ V^T)
    :param X: node attributes of the graph, shape=(n, d)
    :return:
        intraCost: alignment cost based on node attributes, shape=(n, n)
    """

    _, d = X.shape
    X_zero_pos = np.where(np.abs(X).sum(1) == 0)
    X = X / np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    if X_zero_pos[0].shape[0] != 0:
        X[X_zero_pos] = np.sqrt(1/d)
    intraCost = np.exp(-(X @ X.T))

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
    HInd = np.where(H.T == 1)
    crossCost = np.zeros((nx, ny)).astype(np.float64)
    for i in tqdm(range(prodRwrIter), desc="Computing product RWR scores"):
        rwCost_old = crossCost.copy()
        crossCost = (1 + gamma * beta) * nodeCost + (1 - beta) * gamma * L1 @ crossCost @ L2.T
        crossCost[HInd] = 0
        if np.max(np.abs(crossCost - rwCost_old)) < eps:
            break
    crossCost = (1 - gamma) * crossCost
    crossCost[HInd] = 0

    return crossCost
