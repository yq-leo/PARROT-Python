import time
import torch
from utils import *
from tqdm import tqdm


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

    if torch.sum(A1.sum(1) == 0) != 0:
        A1[torch.where(A1.sum(1) == 0)] = torch.ones(nx).int()
    if torch.sum(A2.sum(1) == 0) != 0:
        A2[torch.where(A2.sum(1) == 0)] = torch.ones(ny).int()

    L1 = A1 / A1.sum(1, keepdim=True).to(torch.float64)
    L2 = A2 / A2.sum(1, keepdim=True).to(torch.float64)

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

    # define initial matrix values
    a = torch.ones((nx, 1)).to(torch.float64) / nx
    b = torch.ones((1, ny)).to(torch.float64) / ny
    r = torch.ones((nx, 1)).to(torch.float64) / nx
    c = torch.ones((1, ny)).to(torch.float64) / ny
    l_total = l1 + l2 + l3

    T = torch.ones((nx, ny)).to(torch.float64) / (nx * ny)
    H = H.T + torch.ones((nx, ny)).to(torch.float64) / ny

    # functions for OT
    def mina(H_in, epsilon):
        in_a = torch.ones((nx, 1)).to(torch.float64) / nx
        return -epsilon * torch.log(torch.sum(in_a * torch.exp(-H_in / epsilon), dim=0, keepdim=True))

    def minb(H_in, epsilon):
        in_b = torch.ones((1, ny)).to(torch.float64) / ny
        return -epsilon * torch.log(torch.sum(in_b * torch.exp(-H_in / epsilon), dim=1, keepdim=True))

    def minaa(H_in, epsilon):
        return mina(H_in - torch.min(H_in, dim=0).values.view(1, -1), epsilon) + torch.min(H_in, dim=0).values.view(1, -1)

    def minbb(H_in, epsilon):
        return minb(H_in - torch.min(H_in, dim=1).values.view(-1, 1), epsilon) + torch.min(H_in, dim=1).values.view(-1, 1)

    temp1 = 0.5 * (intraC1 ** 2) @ r @ torch.ones((1, ny)).to(torch.float64) + 0.5 * torch.ones((nx, 1)).to(torch.float64) @ c @ (intraC2 ** 2).T

    resRecord = []
    WRecord = []
    # outIter = min(outIter, int(np.max(crossC) * np.log(max(nx, ny) * (eps ** (-3)))))
    start_time = time.time()
    for i in tqdm(range(outIter), desc="Computing constraint proximal point iteration"):
        T_old = torch.clone(T)
        CGW = temp1 - intraC1 @ T @ intraC2.T
        C = crossC - l2 * torch.log(L1 @ T @ L2.T) - l3 * torch.log(H) + l4 * CGW

        if i == 0:
            C_old = C
        else:
            W_old = torch.sum(T * C_old)
            W = torch.sum(T * C)
            if W <= W_old:
                C_old = C
            else:
                C = C_old

        Q = C - l1 * torch.log(T)
        for j in range(inIter):
            a = minaa(Q - b, l_total)
            b = minbb(Q - a, l_total)
            pass

        T = 0.05 * T_old + 0.95 * r * torch.exp((a + b - Q) / l_total) * c
        res = torch.sum(torch.abs(T - T_old))
        resRecord.append(res)
        WRecord.append(torch.sum(T * C))

    end_time = time.time()
    print(f"Time for optimization: {end_time - start_time:.2f}s")

    return T, WRecord, resRecord






