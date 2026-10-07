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

import gnn_ppo
from gnn_ppo import PPOTrainer, load_graph


def legacy_mixed_candidates(self):
    n=len(self.order)
    scored=[]
    pos={v:i for i,v in enumerate(self.order)}
    for i,v in enumerate(self.order):
        local=max((abs(i-pos[u]) for u in self.G.neighbors(v)),default=0)
        scored.append((local,self.G.degree(v),-i,i))
    scored.sort(reverse=True)
    critical=[item[-1] for item in scored[:min(len(scored),self.critical_nodes)]]

    specs=set()
    for i in critical:
        for offset in self.offsets:
            for sign in (-1,1):
                j=i+sign*offset
                if j<0 or j>=n or j==i:
                    continue
                if self.enable_swap:
                    specs.add((0,i,j))
                if self.enable_relocation:
                    specs.add((1,i,j))
    if not specs and n>=2:
        specs.add((0,0,1))

    self.action_specs=sorted(specs)
    denom=max(1,n-1)
    rows=[]
    for kind,i,j in self.action_specs:
        rows.append([
            float(kind),float(i),float(j),
            abs(j-i)/denom,
            1.0 if j>i else -1.0,
        ])
    return torch.tensor(rows,dtype=torch.float32,device=self.device)


def run(G,seed,episodes,steps,mode):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    trainer=PPOTrainer(
        G,device=torch.device("cpu"),
        critical_nodes=16,
        offsets=(1,2,4,8,16,32),
        enable_swap=True,
        enable_relocation=True,
    )
    if mode=="legacy_mixed":
        trainer.env._build_action_candidates=legacy_mixed_candidates.__get__(
            trainer.env, trainer.env.__class__
        )
    elif mode!="edge_guided":
        raise ValueError(mode)

    start=time.perf_counter()
    result=trainer.train(episodes=episodes,steps=steps,seed=seed)
    elapsed=time.perf_counter()-start
    return result,elapsed


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True)
    p.add_argument("--seeds",type=int,default=10)
    p.add_argument("--episodes",type=int,default=15)
    p.add_argument("--steps",type=int,default=30)
    p.add_argument("--output",required=True)
    p.add_argument("instances",nargs="+")
    a=p.parse_args()

    rows=[]
    for name in a.instances:
        G=load_graph(os.path.join(a.data_dir,name))
        for seed in range(1,a.seeds+1):
            for mode in ("legacy_mixed","edge_guided"):
                result,elapsed=run(G,seed,a.episodes,a.steps,mode)
                rows.append({
                    "instance":name,"seed":seed,"mode":mode,
                    "nodes":G.number_of_nodes(),"edges":G.number_of_edges(),
                    "rcm_bw":result["rcm_bw"],"best_bw":result["best_bw"],
                    "elapsed_s":elapsed,
                })

    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)
    print(f"wrote {len(rows)} rows")


if __name__=="__main__":
    main()
