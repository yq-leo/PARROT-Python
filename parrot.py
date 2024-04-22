import time
import numpy as np
from utils import *


def parrot(A1, A2, X1, X2, H, sepRwrIter, prodRwrIter, alpha, beta, gamma, inIter, outIter, l1, l2, l3, l4):
    """
    Position-aware optimal transport for network alignment.
    :param A1: adjacency matrix of graph G1, shape=(n1, n1)
    :param A2: adjacency matrix of graph G2, shape=(n2, n2)
    :param X1: input node attributes of graph G1, shape=(n1, d)
    :param X2: input node attributes of graph G2, shape=(n2, d)
    :param H: anchor links, shape=(n2, n1)
    :param sepRwrIter: number of iteartion for rwr on separated graphs
    :param prodRwrIter: number of iteration for rwr on product graph
    :param alpha: weight balancing attribute cost and rwr cost
    :param beta: rwr restart ratio
    :param gamma: discounted factor of Bellman equation
    :param inIter: number of inner iteration for logipot algorithm
    :param outIter: number of outer iteration for logipot algorithm
    :param l1: weight for update step regularization
    :param l2: weight for smoothness regularization
    :param l3: weight for prior knowledge regularization
    :param l4: weight balancing Wasserstein and Gromov-Wasserstein distance
    :return:
        T: trasnport plan/alignment score, shape=(n1, n2)
        W: Wasserstein distance along the distance, shape=outIter
        res: diff between two consecutive alignment scores, shape=outIter
    """

    nx, ny = H.T.shape

    if np.sum(A1.sum(1) == 0) != 0:
        A1[np.where(A1.sum(1) == 0)] = np.ones(nx)
    if np.sum(A2.sum(1) == 0) != 0:
        A2[np.where(A2.sum(1) == 0)] = np.ones(ny)

    L1 = A1 / A1.sum(1)
    L2 = A2 / A2.sum(1)

    crossC, intraC1, intraC2 = get_cost(A1, A2, X1, X2, H, sepRwrIter, prodRwrIter, alpha, beta, gamma)
    T, W, res = cpot(L1, L2, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4)

    return T, W, res


def cpot(L1, L2, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4):
    """
    Constraint proximal point iteration for optimal transport.
    :param L1: Laplacian matrix of graph G1, shape=(n1, n1)
    :param L2: Laplacian matrix of graph G2, shape=(n2, n2)
    :param crossC: cross-graph cost matrix, shape=(n1, n2)
    :param intraC1: intra-graph cost matrix for graph 1, shape=(n1, n1)
    :param intraC2: intra-graph cost matrix for graph 2, shape=(n2, n2)
    :param inIter: maximum number of inner iteration
    :param outIter: maximum number of outer iteration
    :param H: anchor links, shape=(n2, n1)
    :param l1: weight for entropy term
    :param l2: weight for smoothness term
    :param l3: weight for preference term
    :param l4: weight for GWD
    :return:
        T: trasnport plan/alignment score, shape=(n1, n2)
        WRecord: Wasserstein distance along the distance, shape=outIter
        resRecord: diff between two consecutive alignment scores, shape=outIter
    """

    nx, ny = crossC.shape
    l4 = l4 * nx * ny
    eps = 0

    # define initial matrix values
    a = np.ones((nx, 1)).astype(np.float64) / nx
    b = np.ones((1, ny)).astype(np.float64) / ny
    r = np.ones((nx, 1)).astype(np.float64) / nx
    c = np.ones((1, ny)).astype(np.float64) / ny
    l = l1 + l2 + l3

    T = np.ones((nx, ny)).astype(np.float64) / (nx * ny)
    H = H.T + np.ones((nx, ny)) / ny

    # functions for OT
    def mina(H_in, epsilon):
        return -epsilon * np.log(np.sum(a * np.exp(-H_in / epsilon), axis=0))

    def minb(H_in, epsilon):
        return -epsilon * np.log(np.sum(b * np.exp(-H_in / epsilon), axis=1))

    def minaa(H_in, epsilon):
        return mina(H_in - np.min(H_in, axis=0).reshape(1, -1), epsilon) + np.min(H_in, axis=0)

    def minbb(H_in, epsilon):
        return minb(H_in - np.min(H_in, axis=1).reshape(-1, 1), epsilon) + np.min(H_in, axis=1)

    temp1 = 0.5 * (intraC1 ** 2) @ r @ np.ones((1, ny)) + 0.5 * np.ones((nx, 1)) @ c @ (intraC2 ** 2)

    resRecord = []
    WRecord = []
    # outIter = min(outIter, int(np.max(crossC) * np.log(max(nx, ny) * (eps ** (-3)))))
    start_time = time.time()
    for i in range(outIter):
        T_old = T.copy()
        CGW = temp1 - intraC1 @ T @ intraC2.T
        C = crossC - l2 * np.log(L1 @ T @ L2.T + eps) - l3 * np.log(H) + l4 * CGW

        if i == 0:
            C_old = C
        else:
            W_old = np.sum(T * C_old)
            W = np.sum(T * C)
            if W <= W_old:
                C_old = C
            else:
                C = C_old

        Q = C - l1 * np.log(T)
        for j in range(inIter):
            a = minaa(Q - b, l).reshape(1, -1)
            b = minbb(Q - a, l).reshape(-1, 1)

        T = 0.05 * T_old + 0.95 * r * np.exp((a + b - Q) / l) * c
        res = np.sum(np.abs(T - T_old))
        resRecord.append(res)
        WRecord.append(np.sum(T * C))

    end_time = time.time()
    print(f"Time for optimization: {end_time - start_time:.2f}s")

    return T, WRecord, resRecord






