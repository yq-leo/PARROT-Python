import numpy as np


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

    sortI = np.argsort(-s, axis=1)
    anchors1, anchors2 = np.where(H.T == 1)
    anchors = np.vstack((anchors1, anchors2)).T
    tests = setdiff(gnd, anchors)
    test_len = tests.shape[0]

    ind = []
    mrr = 0.0
    for i in range(test_len):
        tempInd = np.where(sortI[tests[i, 0]] == tests[i, 1])[0][0]
        ind.append(tempInd)
        mrr += 1 / (tempInd + 1)

    mrr = mrr / test_len

    p = []
    for i in range(len(topK)):
        p.append(np.sum(np.array(ind) < topK[i]))
    p = np.array(p) / test_len

    return p, mrr


def setdiff(a, b):
    """
    Find the difference of two arrays.
    :param a: array 1 (n1 x 2)
    :param b: array 2 (n2 x 2)
    :return: c: difference of a and b (n3 x 2)
    """
    # Use a structured array to perform row-wise set difference
    dtype = np.dtype([('row', np.int64, 2)])  # Define dtype for structured array
    A_struct = a.copy().view(dtype).reshape(-1)  # Convert A
    B_struct = b.copy().view(dtype).reshape(-1)  # Convert B

    # Compute set difference
    C_struct = np.setdiff1d(A_struct, B_struct)
    c = C_struct.view(a.dtype).reshape(-1, 2)  # Convert back to original array form

    return c
