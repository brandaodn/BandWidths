import argparse, csv, os, random, sys, time
import numpy as np
import torch

ROOT=os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from benchmark_ablation import CENTRALITIES, build, eval_random, eval_rcm, eval_centrality
from dqn_v2 import DQNSearchV2

FIXED=["Eigenvector","Katz Centrality"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True)
    p.add_argument("--instance",default="bcsstk01.mtx")
    p.add_argument("--seeds",type=int,default=30)
    p.add_argument("--train-steps",type=int,default=300)
    p.add_argument("--eval-steps",type=int,default=60)
    p.add_argument("--output",required=True)
    a=p.parse_args()

    path=os.path.join(a.data_dir,a.instance)
    nnodes,nedges,G,g,orig,maps=build(path)
    rcm=eval_rcm(G,g,orig)
    total_steps=a.train_steps+a.eval_steps
    rows=[]

    for seed in range(1,a.seeds+1):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        solver=DQNSearchV2(
            graph=g,centrality_maps=maps,centralities=CENTRALITIES,
            initial_bw=orig,train_steps=a.train_steps,eval_steps=a.eval_steps,
            samples_per_action=15,
        )

        t0=time.time()
        train_state,train_hist=solver.train()
        train_time=time.time()-t0

        t1=time.time()
        eval_state,eval_hist=solver.evaluate_greedy()
        eval_time=time.time()-t1

        # matched TRAIN budget
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        random_train=eval_random(g,maps,a.train_steps,seed,orig)
        fixed_train={}
        for name in FIXED:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            fixed_train[name]=eval_centrality(g,maps,name,a.train_steps,orig)

        # matched EVAL budget (policy-quality view; training cost excluded by definition)
        random.seed(seed+10000); np.random.seed(seed+10000); torch.manual_seed(seed+10000)
        random_eval=eval_random(g,maps,a.eval_steps,seed+10000,orig)
        fixed_eval={}
        for name in FIXED:
            random.seed(seed+10000); np.random.seed(seed+10000); torch.manual_seed(seed+10000)
            fixed_eval[name]=eval_centrality(g,maps,name,a.eval_steps,orig)

        # matched END-TO-END budget: DQN used train+evaluation constructions
        random.seed(seed+20000); np.random.seed(seed+20000); torch.manual_seed(seed+20000)
        random_total=eval_random(g,maps,total_steps,seed+20000,orig)
        fixed_total={}
        for name in FIXED:
            random.seed(seed+20000); np.random.seed(seed+20000); torch.manual_seed(seed+20000)
            fixed_total[name]=eval_centrality(g,maps,name,total_steps,orig)

        dqn_total=min(train_state.best_bw,eval_state.best_bw)

        row={
            "instance":a.instance,"seed":seed,"nodes":nnodes,"edges":nedges,
            "original_bw":orig,"rcm_bw":rcm,
            "train_steps":a.train_steps,"eval_steps":a.eval_steps,"total_steps":total_steps,
            "samples_per_action":15,
            "epsilon_final":solver.agent.epsilon,
            "dqn_train_best":int(train_state.best_bw),
            "random_train_best":int(random_train),
            "eigen_train_best":int(fixed_train["Eigenvector"]),
            "katz_train_best":int(fixed_train["Katz Centrality"]),
            "dqn_eval_best":int(eval_state.best_bw),
            "random_eval_best":int(random_eval),
            "eigen_eval_best":int(fixed_eval["Eigenvector"]),
            "katz_eval_best":int(fixed_eval["Katz Centrality"]),
            "dqn_total_best":int(dqn_total),
            "random_total_best":int(random_total),
            "eigen_total_best":int(fixed_total["Eigenvector"]),
            "katz_total_best":int(fixed_total["Katz Centrality"]),
            "train_time_s":train_time,"eval_time_s":eval_time,
        }

        # Record action frequencies in last 20% training and all greedy evaluation.
        names=solver.names
        tail=train_hist[int(0.8*len(train_hist)):]
        for name in names:
            key=name.lower().replace(" ","_")
            row[f"tail_{key}"]=sum(x["centrality"]==name for x in tail)/max(1,len(tail))
            row[f"eval_{key}"]=sum(x["centrality"]==name for x in eval_hist)/max(1,len(eval_hist))
        rows.append(row)
        print(seed,"train",row["dqn_train_best"],row["random_train_best"],
              "eval",row["dqn_eval_best"],row["random_eval_best"],
              "total",row["dqn_total_best"],row["random_total_best"],
              "eig",row["eigen_total_best"],"katz",row["katz_total_best"])

    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)

if __name__=="__main__": main()
