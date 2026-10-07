import argparse, csv, os, random, sys, time
from collections import Counter
import numpy as np
import torch
import networkx as nx

ROOT=os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from benchmark_ablation import CENTRALITIES, build, eval_random, eval_rcm
from dqn_v2 import DQNSearchV2


def summarize_actions(history, names, tail_fraction=0.2):
    if not history:
        return {}
    start = int(len(history) * (1.0 - tail_fraction))
    counts = Counter(row["centrality"] for row in history[start:])
    total = max(1, sum(counts.values()))
    return {name: counts.get(name, 0) / total for name in names}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--eval-steps", type=int, default=60)
    p.add_argument("--output", required=True)
    p.add_argument("instances", nargs="+")
    a=p.parse_args()

    rows=[]
    for fname in a.instances:
        path=os.path.join(a.data_dir, fname)
        nnodes,nedges,G,g,orig,maps=build(path)
        rcm=eval_rcm(G,g,orig)

        for seed in range(1, a.seeds+1):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            solver=DQNSearchV2(
                graph=g, centrality_maps=maps, centralities=CENTRALITIES,
                initial_bw=orig, train_steps=a.train_steps,
                eval_steps=a.eval_steps, samples_per_action=15,
            )

            t0=time.time()
            train_state, train_hist=solver.train()
            train_time=time.time()-t0

            t1=time.time()
            eval_state, eval_hist=solver.evaluate_greedy()
            eval_time=time.time()-t1

            # Fair search-budget baselines for the separate train and eval phases.
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            random_train=eval_random(g,maps,a.train_steps,seed,orig)
            random.seed(seed+10000); np.random.seed(seed+10000); torch.manual_seed(seed+10000)
            random_eval=eval_random(g,maps,a.eval_steps,seed+10000,orig)

            tail=summarize_actions(train_hist, solver.names, 0.2)
            eval_dist=summarize_actions(eval_hist, solver.names, 1.0)
            row={
                "instance":fname,"seed":seed,"nodes":nnodes,"edges":nedges,
                "original_bw":orig,"rcm_bw":rcm,
                "train_steps":a.train_steps,"eval_steps":a.eval_steps,
                "epsilon_final":solver.agent.epsilon,
                "dqn_train_best":int(train_state.best_bw),
                "dqn_eval_best":int(eval_state.best_bw),
                "random_train_best":int(random_train),
                "random_eval_best":int(random_eval),
                "train_time_s":train_time,"eval_time_s":eval_time,
                "train_improvements":int(sum(x["improved"] for x in train_hist)),
                "eval_improvements":int(sum(x["improved"] for x in eval_hist)),
            }
            for name in solver.names:
                key=name.lower().replace(" ","_")
                row[f"tail_{key}"]=tail.get(name,0.0)
                row[f"eval_{key}"]=eval_dist.get(name,0.0)
            rows.append(row)
            print(fname, seed, "orig", orig, "train", train_state.best_bw,
                  "greedy", eval_state.best_bw, "random", random_train,
                  "eps", solver.agent.epsilon)

    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__=="__main__":
    main()
