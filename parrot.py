import time
import torch
from utils import *
from tqdm import tqdm
import numpy as np
import random
import os


def parrot(dataset_name, A1, A2, X1, X2, H, sepRwrIter, prodRwrIter, alpha, beta, gamma, inIter, outIter, l1, l2, l3, l4,
           no_joint_rwr=False, use_pgna=False, use_num=False, use_pgna_num=False, self_train='off', gnd=None, time_recorder=None):
    """
    Position-aware optimal transport for network alignment.
    :param dataset_name: dataset name
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
    :param no_joint_rwr: whether to use joint rwr
    :param use_pgna: whether to use PGNA embeddings
    :param use_num: whether to use non-uniform marginal distribution
    :param use_pgna_num: whether to use PGNA embeddings for non-uniform marginal distribution
    :param self_train: whether to use self-training
    :param gnd: ground truth alignment (training + test pairs)
    :param time_recorder: time recorder
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

    crossC, intraC1, intraC2 = get_cost(dataset_name, A1, A2, X1, X2, H, sepRwrIter, prodRwrIter, alpha, beta, gamma, no_joint_rwr, time_recorder)

    if use_pgna:
        pgna_emb = np.load(f"outputs/pgna_constrained/pgna_{dataset_name}_embeddings.npz")
        emb1 = torch.from_numpy(pgna_emb['out1']).to(torch.float64)
        emb2 = torch.from_numpy(pgna_emb['out2']).to(torch.float64)
        # crossC = 0.95 * torch.exp(-(emb1 @ emb2.T)) + 0.05 * crossC
        crossC = torch.exp(-(emb1 @ emb2.T))

    u = torch.ones(nx).to(torch.float64) / nx
    v = torch.ones(ny).to(torch.float64) / ny
    if use_num:
        if use_pgna_num:
            pgna_emb = np.load(f"outputs/pgna_constrained/pgna_{dataset_name}_embeddings.npz")
            emb1 = torch.from_numpy(pgna_emb['out1']).to(torch.float64)
            emb2 = torch.from_numpy(pgna_emb['out2']).to(torch.float64)
            sim = torch.exp(emb1 @ emb2.T)
            norm_cost = sim / torch.sum(sim)
        else:
            norm_cost = crossC / torch.sum(crossC)
        u = norm_cost.sum(1)
        v = norm_cost.sum(0)

    # T, W, res = cpot_new(L1, L2, u, v, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4,
    #                      self_train=self_train, gnd=gnd)
    start_time = time.time()
    T, W, res = cpot(L1, L2, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4)
    end_time = time.time()
    time_recorder['cpot'].append(end_time - start_time)
    # T, W, res = cpot_org(L1, L2, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4)

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
    b = torch.ones((nx, 1)).to(torch.float64) / nx
    a = torch.ones((1, ny)).to(torch.float64) / ny
    r = torch.ones((nx, 1)).to(torch.float64) / nx
    c = torch.ones((1, ny)).to(torch.float64) / ny
    l_total = l1 + l2 + l3

    T = torch.ones((nx, ny)).to(torch.float64) / (nx * ny)
    H = H.T + torch.ones((nx, ny)).to(torch.float64) / (nx * ny)

    # functions for OT
    def mina(H_in, epsilon):
        in_a = torch.ones((1, nx)).to(torch.float64) / nx
        return -epsilon * torch.log(in_a @ torch.exp(-H_in / epsilon))

    def minb(H_in, epsilon):
        in_b = torch.ones((ny, 1)).to(torch.float64) / ny
        return -epsilon * torch.log(torch.exp(-H_in / epsilon) @ in_b)

    def minaa(H_in, epsilon):
        return mina(H_in - torch.min(H_in, dim=0).values.view(1, -1), epsilon) + torch.min(H_in, dim=0).values.view(1, -1)

    def minbb(H_in, epsilon):
        return minb(H_in - torch.min(H_in, dim=1).values.view(-1, 1), epsilon) + torch.min(H_in, dim=1).values.view(-1, 1)

    temp1 = 0.5 * (intraC1 ** 2) @ r @ torch.ones((1, ny)).to(torch.float64) + 0.5 * torch.ones((nx, 1)).to(torch.float64) @ c @ (intraC2 ** 2).T

    resRecord = []
    WRecord = []
    # outIter = min(outIter, int(np.max(crossC) * np.log(max(nx, ny) * (eps ** (-3)))))
    start_time = time.time()
    for i in range(outIter):
        T_old = torch.clone(T)
        CGW = temp1 - intraC1 @ T @ intraC2.T
        C = crossC - l2 * torch.log(L1.T @ T @ L2) - l3 * torch.log(H) + l4 * CGW

        wasserstein = torch.sum(crossC * T)
        edge_loss = torch.sum(CGW * T)
        neigh_loss = -torch.sum(torch.log(L1.T @ T @ L2) * T) + torch.sum(torch.log(T) * T)
        align_loss = -torch.sum(torch.log(H) * T) + torch.sum(torch.log(T) * T)
        entropy = -torch.sum(T * torch.log(T))
        print(f"Iter {i}: wasserstein={wasserstein:.6f}, edge={edge_loss:.6f}, neigh={neigh_loss:.6f}, align={align_loss:.6f}, entropy={entropy:.6f},"
              f"s={torch.sum(T):.6f}, s_hat={torch.sum(L1.T @ T @ L2):.6f}")

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

        T = 0.05 * T_old + 0.95 * r * torch.exp((a + b - Q) / l_total) * c
        res = torch.sum(torch.abs(T - T_old))
        resRecord.append(res)
        WRecord.append(torch.sum(T * C))

    end_time = time.time()
    print(f"Time for optimization: {end_time - start_time:.2f}s")

    return T, WRecord, resRecord


def cpot_new(L1, L2, u, v, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4, self_train='off', gnd=None):
    """
    Constraint proximal point iteration for optimal transport.
    :param L1: Laplacian matrix of graph G1, shape=(n1, n1)
    :param L2: Laplacian matrix of graph G2, shape=(n2, n2)
    :param u: marginal distribution of graph 1, shape=(n1, 1)
    :param v: marginal distribution of graph 2, shape=(n2, 1)
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
    :param self_train: whether to use self-training
    :param gnd: ground truth alignment (training + test pairs)
    :return:
        T: trasnport plan/alignment score, shape=(n1, n2)
        WRecord: Wasserstein distance along the distance, shape=outIter
        resRecord: diff between two consecutive alignment scores, shape=outIter
    """

    lambda_p = l1
    lambda_n = l2
    lambda_a = l3
    lambda_e = l4

    nx, ny = crossC.shape
    lambda_e *= nx * ny

    # define initial matrix values
    a = torch.ones(nx).to(torch.float64) / nx
    b = torch.ones(ny).to(torch.float64) / ny
    lambda_total = lambda_p + lambda_n + lambda_a

    S = torch.ones((nx, ny)).to(torch.float64) / (nx * ny)
    H = H.T + torch.ones((nx, ny)).to(torch.float64) / (nx * ny)

    gnd_dict = {int(node1): int(node2) for node1, node2 in gnd}

    temp1 = 0.5 * (intraC1 ** 2) @ u.view(-1, 1) @ torch.ones((1, ny)).to(torch.float64) + 0.5 * torch.ones((nx, 1)).to(torch.float64) @ v.view(1, -1) @ (intraC2 ** 2).T

    resRecord = []
    WRecord = []
    num_anchors = torch.where(H > 1)[0].shape[0]
    start_time = time.time()
    for i in range(outIter):
        hit_rate = 0

        # if i > 0:
        #     rank = torch.argsort(-S, dim=1)
        #     rank_max = torch.argsort(-S.max(dim=1)[0] * (1 - torch.sigmoid(-torch.sum(S * torch.log(S), dim=1))))
        #     rank_entropy = torch.argsort(-torch.sum(S * torch.log(S), dim=1))
        #     rank_rwr = torch.argsort(crossC[torch.arange(nx), rank[:, 0]])
        #
        #     ord_max = torch.zeros(nx).to(torch.int64)
        #     ord_entropy = torch.zeros(nx).to(torch.int64)
        #     ord_rwr = torch.zeros(nx).to(torch.int64)
        #     for j in range(nx):
        #         ord_max[rank_max[j]] = j + 1
        #         ord_entropy[rank_entropy[j]] = j + 1
        #         ord_rwr[rank_rwr[j]] = j + 1
        #     # ord_mean = ((outIter-i)/outIter * ord_max + (i/outIter) * ord_entropy) / 2
        #     ord_mean = ord_max
        #     pred = torch.argsort(ord_mean)
        #
        #     num_add = num_anchors // outIter // 4
        #     add = num_add
        #     hit_rate = 0
        #     for j in range(nx):
        #         idx1 = pred[j].item()
        #         idx2 = rank[idx1, 0].item()
        #         if add <= 0:
        #             break
        #         if H[idx1, idx2] < 1:
        #             if idx1 in gnd_dict and gnd_dict[idx1] == idx2:
        #                 hit_rate += 1
        #             crossC[idx1, idx2] = 0
        #             H[idx1, idx2] += 1
        #             add -= 1
        #     hit_rate /= num_add

        # God mode anchor selection
        if self_train == 'god':
            if i > 0:
                add = 0
                pred_anchors1, pred_anchors2 = [], []
                while add < num_anchors // outIter:
                    idx1 = random.randint(0, nx - 1)
                    if idx1 not in gnd_dict:
                        continue
                    idx2 = gnd_dict[idx1]
                    if H[idx1][idx2] < 1:
                        pred_anchors1.append(idx1)
                        pred_anchors2.append(idx2)
                        add += 1
                pred_anchors1 = torch.tensor(pred_anchors1).to(torch.int64)
                pred_anchors2 = torch.tensor(pred_anchors2).to(torch.int64)
                crossC[pred_anchors1, pred_anchors2] = 0
                H[pred_anchors1, pred_anchors2] += 1

        # God mode anchor selection (only from hit nodes)
        elif self_train == 'hit':
            if i > 0:
                rank = torch.argsort(-S, dim=1)
                truth = -1 * torch.ones(nx).to(torch.int64)
                truth[gnd[:, 0]] = gnd[:, 1]
                hit_nodes = torch.where(rank[:, 0] == truth)[0].numpy().tolist()
                add = 0
                while add < num_anchors // outIter:
                    idx1 = random.choice(hit_nodes)
                    idx2 = gnd_dict[idx1]
                    if H[idx1][idx2] < 1:
                        crossC[idx1, idx2] = 0
                        H[idx1, idx2] += 1
                        add += 1

        S_old = torch.clone(S)
        L = temp1 - intraC1 @ S @ intraC2.T
        C = crossC - lambda_n * torch.log(L1.T @ S @ L2) - lambda_a * torch.log(H) + lambda_e * L

        wassserstein = torch.sum(crossC * S)
        edge_loss = torch.sum(L * S)
        neigh_loss = -torch.sum(torch.log(L1.T @ S @ L2) * S) + torch.sum(torch.log(S) * S)
        align_loss = -torch.sum(torch.log(H) * S) + torch.sum(torch.log(S) * S)
        entropy = -torch.sum(S * torch.log(S))
        print(
            f"Iter {i + 1}: wasserstein={wassserstein: .6f}, edge={edge_loss:.8f}, neigh={neigh_loss:.6f}, align={align_loss:.6f}, entropy={entropy:.6f}, "
            f"s={torch.sum(S):.6f}, s_hat={torch.sum(L1.T @ S @ L2):.6f}, hit_rate={hit_rate:.3f}")

        if i == 0:
            C_old = C
        else:
            W_old = torch.sum(S * C_old)
            W = torch.sum(S * C)
            if W <= W_old:
                C_old = C
            else:
                C = C_old

        Q = C - lambda_p * torch.log(S)
        for j in range(inIter):
            a = -lambda_total * torch.log(torch.exp((b.view(1, -1) - Q) / lambda_total).sum(1) / u)
            b = -lambda_total * torch.log(torch.exp((a.view(1, -1) - Q.T) / lambda_total).sum(1) / v)

        a, b = a.view((-1, 1)), b.view((-1, 1))
        S = 0.05 * S_old + 0.95 * torch.exp((a + b.T - Q) / lambda_total)

        res = torch.sum(torch.abs(S - S_old))
        resRecord.append(res)
        WRecord.append(torch.sum(S * C))

    end_time = time.time()
    print(f"Time for optimization: {end_time - start_time:.2f}s")

    return S, resRecord, WRecord


def cpot_org(L1, L2, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4):
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

    lambda_p = l1
    lambda_n = l2
    lambda_a = l3
    lambda_e = l4

    nx, ny = crossC.shape
    lambda_e *= nx * ny

    # define initial matrix values
    a = torch.ones(nx).to(torch.float64) / nx
    b = torch.ones(ny).to(torch.float64) / ny
    u = torch.ones(nx).to(torch.float64) / nx
    v = torch.ones(ny).to(torch.float64) / ny
    lambda_total = lambda_p + lambda_n + lambda_a

    S = torch.ones((nx, ny)).to(torch.float64) / (nx * ny)
    H = H.T + torch.ones((nx, ny)).to(torch.float64) / (nx * ny)

    temp1 = 0.5 * (intraC1 ** 2) @ u.view(-1, 1) @ torch.ones((1, ny)).to(torch.float64) + 0.5 * torch.ones((nx, 1)).to(torch.float64) @ v.view(1, -1) @ (intraC2 ** 2).T

    start_time = time.time()
    for i in tqdm(range(outIter), desc="Computing constraint proximal point iteration"):
        S_old = torch.clone(S)
        L = temp1 - intraC1 @ S @ intraC2.T
        C = crossC - 0 * torch.log(L1.T @ S @ L2) - 0 * torch.log(H) + 0 * L

        # if i == 0:
        #     C_old = C
        # else:
        #     W_old = torch.sum(S * C_old)
        #     W = torch.sum(S * C)
        #     if W <= W_old:
        #         C_old = C
        #     else:
        #         C = C_old

        Q = C - 0 * torch.log(S)
        K = torch.exp(-Q / lambda_total)
        for j in range(inIter):
            a = u / (K @ b)
            b = v / (K.T @ a)

        S = 0.05 * S_old + 0.95 * torch.diag(a) @ K @ torch.diag(b)
    end_time = time.time()
    print(f"Time for optimization: {end_time - start_time:.2f}s")

    return S, [], []

