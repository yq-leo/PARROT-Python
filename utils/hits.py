import torch


def get_hits(s, gnd, H, topK):
    """
    Calculate Hits@K and MRR
    :param s: alignment score
    :param gnd: ground truth alignment matrix
    :param H: anchor links, shape=(n2, n1)
    :param topK: top-K list
    :return:
        p: Hits@K
        mrr: Mean Reciprocal Rank
    """

    sortI = torch.argsort(-s, dim=1)
    anchors1, anchors2 = torch.where(H.T == 1)
    anchors = torch.vstack((anchors1, anchors2)).T
    tests = setdiff(gnd, anchors)
    test_len = tests.shape[0]

    ind = []
    mrr = 0.0
    for i in range(test_len):
        tempInd = torch.where(sortI[tests[i, 0]] == tests[i, 1])[0][0]
        ind.append(tempInd)
        mrr += 1 / (tempInd + 1)

    mrr = mrr / test_len

    p = []
    for i in range(len(topK)):
        p.append(torch.sum(torch.tensor(ind) < topK[i]))
    p = torch.tensor(p).to(torch.float64) / test_len

    return p, mrr


# def setdiff(a, b):
#     """
#     Find the difference of two arrays.
#     :param a: array 1 (n1 x 2)
#     :param b: array 2 (n2 x 2)
#     :return: c: difference of a and b (n3 x 2)
#     """
#     # Use a structured array to perform row-wise set difference
#     dtype = np.dtype([('row', np.int64, 2)])  # Define dtype for structured array
#     A_struct = a.copy().view(dtype).reshape(-1)  # Convert A
#     B_struct = b.copy().view(dtype).reshape(-1)  # Convert B
#
#     # Compute set difference
#     C_struct = np.setdiff1d(A_struct, B_struct)
#     c = C_struct.view(a.dtype).reshape(-1, 2)  # Convert back to original array form
#
#     return c

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
