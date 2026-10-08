import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch

ROOT=os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)

from gnn_ppo import load_graph,PPOTrainer


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True)
    p.add_argument("--seeds",type=int,default=10)
    p.add_argument("--episodes",type=int,default=10)
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--output",required=True)
    p.add_argument("instances",nargs="+")
    a=p.parse_args()

    rows=[]
    for name in a.instances:
        G=load_graph(os.path.join(a.data_dir,name))
        for seed in range(1,a.seeds+1):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            trainer=PPOTrainer(G,device=torch.device("cpu"))
            start=time.perf_counter()
            result=trainer.train(episodes=a.episodes,steps=a.steps,seed=seed)
            elapsed=time.perf_counter()-start
            rows.append({
                "instance":name,
                "seed":seed,
                "nodes":G.number_of_nodes(),
                "edges":G.number_of_edges(),
                "rcm_bw":result["rcm_bw"],
                "ppo_best_bw":result["best_bw"],
                "improvement":result["rcm_bw"]-result["best_bw"],
                "improvement_pct":0.0 if result["rcm_bw"]==0 else 100.0*(result["rcm_bw"]-result["best_bw"])/result["rcm_bw"],
                "elapsed_s":elapsed,
            })

    fields=list(rows[0].keys())
    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows")


if __name__=="__main__":
    main()
