import argparse, csv, os, random, sys, time
import numpy as np
import torch

ROOT=os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from benchmark_ablation import CENTRALITIES, build
from centralities import centrality_heuristic
from dqn_v2 import DQNSearchV2
from reviewer_experiments import pseudo_peripheral_node, raw_rcm

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def run_dqn(g,maps,orig,train_steps,eval_steps,seed,start_node):
    seed_all(seed)
    solver=DQNSearchV2(g,maps,CENTRALITIES,orig,train_steps,eval_steps,15,start_node=start_node)
    t0=time.perf_counter()
    tr,_=solver.train(); ev,_=solver.evaluate_greedy()
    return int(min(tr.best_bw,ev.best_bw)), time.perf_counter()-t0

def run_random(g,maps,orig,decisions,seed,start_node):
    # Dedicated action RNG keeps the centrality sequence independent from construction RNG.
    action_rng=random.Random(seed+314159)
    seed_all(seed)
    names=list(CENTRALITIES)
    best=int(orig)
    t0=time.perf_counter()
    for _ in range(decisions):
        name=action_rng.choice(names)
        cost,_,_,_=centrality_heuristic(
            graph=g,centrality_values=maps[name],cent_str=name,
            alpha=0.3,iter_max=15,centralities=CENTRALITIES,start_node=start_node)
        best=min(best,int(cost))
    return best,time.perf_counter()-t0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True)
    p.add_argument("--instance",required=True)
    p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--seed-start",type=int,default=1)
    p.add_argument("--train-steps",type=int,default=300)
    p.add_argument("--eval-steps",type=int,default=60)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    path=os.path.join(a.data_dir,a.instance)
    n,m,G,g,orig,maps=build(path)
    pp=pseudo_peripheral_node(G)
    rcm=raw_rcm(G,g)
    decisions=a.train_steps+a.eval_steps
    rows=[]
    for seed in range(a.seed_start,a.seed_start+a.seeds):
        ds,tds=run_dqn(g,maps,orig,a.train_steps,a.eval_steps,seed,None)
        dp,tdp=run_dqn(g,maps,orig,a.train_steps,a.eval_steps,seed,pp)
        rs,trs=run_random(g,maps,orig,decisions,seed,None)
        rp,trp=run_random(g,maps,orig,decisions,seed,pp)
        row=dict(instance=a.instance,seed=seed,nodes=n,edges=m,original_bw=orig,
                 raw_rcm_bw=rcm,pseudo_peripheral_node=pp,
                 dqn_standard=ds,dqn_peripheral=dp,
                 random_standard=rs,random_peripheral=rp,
                 dqn_standard_time_s=tds,dqn_peripheral_time_s=tdp,
                 random_standard_time_s=trs,random_peripheral_time_s=trp)
        rows.append(row)
        print(a.instance,seed,"DQN",ds,dp,"Random",rs,rp,"RCM",rcm)
    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

if __name__=="__main__": main()
