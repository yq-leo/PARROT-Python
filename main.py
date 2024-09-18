import json
from utils import *
from args import *
from parrot import parrot
import csv
import os
import numpy as np
import torch


if __name__ == "__main__":
    args = make_args()
    with open(f"settings/{args.dataset}.json", "r") as f:
        settings = json.load(f)

    graph1, graph2 = settings["graph1"], settings["graph2"]
    use_attr = settings["use_attr"]
    rwrIter = settings["rwrIter"]
    rwIter = settings["rwIter"]
    alpha = settings["alpha"]
    beta = settings["beta"]
    gamma = settings["gamma"]
    inIter = settings["inIter"] if args.inIter < 0 else args.inIter
    outIter = settings["outIter"] if args.outIter < 0 else args.outIter
    l1 = settings["l1"]
    l2 = settings["l2"] if not args.no_neigh_reg else 0
    l3 = settings["l3"] if not args.no_pref_reg else 0
    l4 = settings["l4"] if not args.no_edge_reg else 0

    adj1, adj2, x1, x2, gnd, H = load_data(f"datasets/{args.dataset}.mat", graph1, graph2, use_attr, shuffle=args.shuffle)
    # pgna_emb = np.load(f"outputs/pgna_constrained/pgna_{args.dataset}_embeddings.npz")
    # x1 = torch.from_numpy(pgna_emb['out1']).to(torch.float64)
    # x2 = torch.from_numpy(pgna_emb['out2']).to(torch.float64)

    print(f"Graph 1: {adj1.shape}, {x1.shape if x1 is not None else None}")
    print(f"Graph 2: {adj2.shape}, {x2.shape if x2 is not None else None}")
    print(f"Ground truth: {gnd.shape}, {H.shape}")

    S, W, res = parrot(args.dataset, adj1, adj2, x1, x2, H, rwrIter, rwIter, alpha, beta, gamma, inIter, outIter, l1, l2, l3, l4,
                       args.no_joint_rwr, args.use_pgna, args.use_num, args.use_pgna_num, args.self_train, gnd)
    train_p, train_mrr, test_p, test_mrr = get_hits(S, gnd, H, settings["topK"])

    # print(f"Training results:")
    # for i in range(len(train_p)):
    #     print(f"Top-{settings['topK'][i]}: {train_p[i]:.3f}")
    # print(f"MRR: {train_mrr:.3f}")

    print(f"Test results:")
    for i in range(len(test_p)):
        print(f"Top-{settings['topK'][i]}: {test_p[i]:.3f}")
    print(f"MRR: {test_mrr:.3f}")

    if args.record:
        exp_name = "self_train"
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists(f"results/{exp_name}_test.csv"):
            with open(f"results/{exp_name}_test.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([""] + [f"Hit@{k}" for k in settings["topK"]] + ["MRR"])

        with open(f"results/{exp_name}_test.csv", "a", newline='') as f:
            writer = csv.writer(f)
            header = f"{args.dataset}_({args.self_train})"
            writer.writerow([header] + [f"{p:.3f}" for p in test_p] + [f"{test_mrr:.3f}"])
