import json
from utils import *
from args import *
from parrot import parrot
import csv
import os
import numpy as np
import torch
import time


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

    adj1_org, adj2_org, x1_org, x2_org, gnd_org, H_org = load_data(f"datasets/{args.dataset}.mat", graph1, graph2, use_attr, shuffle=args.shuffle)

    test_p_list = []
    test_mrr_list = []
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")
        # add edge noise to the target graph
        start_time = time.time()
        adj1 = torch.clone(adj1_org)
        adj2 = perturb_edges(torch.clone(adj2_org), args.edge_noise)
        end_time = time.time()
        print(f"Time for edge perturbation: {end_time - start_time:.3f}s")
        start_time = time.time()
        if x1_org is not None and x2_org is not None:
            x1 = torch.clone(x1_org)
            x2 = perturb_attr(torch.clone(x2_org), args.attr_noise, args.strong_noise)
        else:
            x1 = None
            x2 = None
        end_time = time.time()
        print(f"Time for attribute perturbation: {end_time - start_time:.3f}s")

        gnd = torch.clone(gnd_org)
        H = torch.clone(H_org)

        print(f"Graph 1: {adj1.shape}, {x1.shape if x1 is not None else None}")
        print(f"Graph 2: {adj2.shape}, {x2.shape if x2 is not None else None}")
        print(f"Ground truth: {gnd.shape}, {H.shape}")

        S, W, res = parrot(args.dataset, adj1, adj2, x1, x2, H, rwrIter, rwIter, alpha, beta, gamma, inIter, outIter, l1, l2, l3, l4,
                           args.no_joint_rwr, args.use_pgna, args.use_num, args.use_pgna_num, args.self_train, gnd, args.edge_noise, args.attr_noise)
        train_p, train_mrr, test_p, test_mrr = get_hits(S, gnd, H, settings["topK"])

        print(f"Test results:")
        for i in range(len(test_p)):
            print(f"Top-{settings['topK'][i]}: {test_p[i]:.3f}")
        print(f"MRR: {test_mrr:.3f}")

        test_p_list.append(test_p.numpy())
        test_mrr_list.append(test_mrr.item())

    if args.robust:
        test_p_list = rm_out(np.sort(np.array(test_p_list), axis=0))
        test_mrr_list = rm_out(np.sort(np.array(test_mrr_list)))
    test_p = np.mean(test_p_list, axis=0)
    test_p_std = np.std(test_p_list, axis=0)
    test_mrr = np.mean(test_mrr_list)
    test_mrr_std = np.std(test_mrr_list)

    if args.record:
        exp_name = args.exp_name
        if not os.path.exists("results"):
            os.makedirs("results")
        out_path = f"results/{exp_name}_results.csv"
        if not os.path.exists(out_path):
            with open(out_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([""] + [f"Hit@{k}" for k in settings["topK"]] + ["MRR"] + [f"std@{k}" for k in settings["topK"]] + ["std_MRR"])

        with open(out_path, "a", newline='') as f:
            writer = csv.writer(f)
            if exp_name == "edge_noise":
                header = f"{args.dataset}_({args.edge_noise:.1f}{'_robust' if args.robust else ''}{'_strong' if args.strong_noise else ''})"
            else:
                header = f"{args.dataset}_({args.attr_noise:.1f}{'_robust' if args.robust else ''}{'_strong' if args.strong_noise else ''})"
            writer.writerow([header] + [f"{p:.3f}" for p in test_p] + [f"{test_mrr:.3f}"] + [f"{p:.3f}" for p in test_p_std] + [f"{test_mrr_std:.3f}"])
