import torch
from torch.nn.functional import softmax

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

    s1 = s / s.sum(1, keepdim=True)
    s2 = s / s.sum(0, keepdim=True)
    entropy1 = -torch.sum(s1 * torch.log(s1), dim=1)
    entropy2 = -torch.sum(s2 * torch.log(s2), dim=1)

    print(f"Entropy1: min={entropy1.min().item():.4f}, max={entropy1.max().item():.4f}, mean={entropy1.mean().item():.4f}")
    print(f"Entropy2: min={entropy2.min().item():.4f}, max={entropy2.max().item():.4f}, mean={entropy2.mean().item():.4f}")

    sortI = torch.argsort(-s, dim=1)

    anchors1, anchors2 = torch.where(H.T == 1)
    anchors = torch.vstack((anchors1, anchors2)).T
    tests = setdiff(gnd, anchors)
    test_len = tests.shape[0]

    hit1 = sortI[tests[:, 0], 0].numpy().tolist()
    print(f"Upperbound: {len(set(hit1)) / test_len:.4f}")

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
