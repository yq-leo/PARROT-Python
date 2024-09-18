import torch


def get_hits(s, gnd, H, topK):
    """
    Calculate Hits@K and MRR
    :param s: alignment score
    :param gnd: ground truth alignment matrix
    :param H: anchor links, shape=(n2, n1)
    :param topK: top-K list
    :return:
        train_p: Hits@K for training pairs
        train_mrr: Mean Reciprocal Rank for training pairs
        test_p: Hits@K for test pairs
        test_mrr: Mean Reciprocal Rank for test pairs
    """

    sortI = torch.argsort(-s, dim=1)

    anchors1, anchors2 = torch.where(H.T == 1)
    anchors = torch.vstack((anchors1, anchors2)).T
    tests = setdiff(gnd, anchors)

    num_train, num_test = anchors.shape[0], tests.shape[0]

    train_ind = []
    train_mrr = 0.0
    for i in range(num_train):
        tempInd = torch.where(sortI[anchors[i, 0]] == anchors[i, 1])[0][0]
        train_ind.append(tempInd)
        train_mrr += 1 / (tempInd + 1)

    train_mrr = train_mrr / num_train

    train_p = []
    for i in range(len(topK)):
        train_p.append(torch.sum(torch.tensor(train_ind) < topK[i]))
    train_p = torch.tensor(train_p).to(torch.float64) / num_train

    test_ind = []
    test_mrr = 0.0
    for i in range(num_test):
        tempInd = torch.where(sortI[tests[i, 0]] == tests[i, 1])[0][0]
        test_ind.append(tempInd)
        test_mrr += 1 / (tempInd + 1)

    test_mrr = test_mrr / num_test

    test_p = []
    for i in range(len(topK)):
        test_p.append(torch.sum(torch.tensor(test_ind) < topK[i]))
    test_p = torch.tensor(test_p).to(torch.float64) / num_test

    return train_p, train_mrr, test_p, test_mrr


def setdiff(a, b):
    """
    Find the difference of two tensors.
    :param a: tensor 1 (n1 x 2)
    :param b: tensor 2 (n2 x 2)
    :return: c: difference of a and b (n3 x 2)
    """
    # Use a structured array to perform row-wise set difference
    a_view = a.view(-1, 1, 2)  # Convert a(tensor) to (n1, 1, 2)
    b_view = b.view(1, -1, 2)  # Convert b(tensor) to (1, n2, 2)

    # Compare each element of a with each element of b
    mask = torch.all(a_view != b_view, dim=2).all(dim=1)

    # Select elements in a that are not in b
    c = a[mask]

    return c
