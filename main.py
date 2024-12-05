import json
from utils import *
from args import *
from parrot import parrot
import numpy as np
import os
import csv


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
    inIter = settings["inIter"]
    outIter = settings["outIter"]
    l1 = settings["l1"]
    l2 = settings["l2"]
    l3 = settings["l3"]
    l4 = settings["l4"]

    rwr_time_list = []
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")

        adj1, adj2, x1, x2, gnd, H = load_data(f"datasets/{args.dataset}.mat", graph1, graph2, use_attr)
        print(f"Graph 1: {adj1.shape}, {x1.shape if x1 is not None else None}")
        print(f"Graph 2: {adj2.shape}, {x2.shape if x2 is not None else None}")
        print(f"Ground truth: {gnd.shape}, {H.shape}")

        S, W, res = parrot(args.dataset, adj1, adj2, x1, x2, H, rwrIter, rwIter, alpha, beta, gamma, inIter, outIter, l1, l2, l3, l4, rwr_time_list)
        p, mrr = get_hits(S, gnd, H, settings["topK"])
        for i in range(len(p)):
            print(f"Top-{settings['topK'][i]}: {p[i]:.3f}")
        print(f"MRR: {mrr:.3f}")

    print(f"Average RWR time: {np.mean(rwr_time_list):.4f}")

    if args.record:
        exp_name = "times"
        if not os.path.exists("results"):
            os.makedirs("results")
        out_file = f"results/{exp_name}.csv"
        if not os.path.exists(out_file):
            with open(out_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Dataset", "RWR_time"])

        with open(out_file, "a", newline='') as f:
            writer = csv.writer(f)
            header = f"{args.dataset}"
            writer.writerow([header, np.mean(rwr_time_list)])
