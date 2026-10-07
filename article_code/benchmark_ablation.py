import argparse, csv, os, random, sys, time
import numpy as np
import torch
import networkx as nx

ROOT=os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from agent import Agent
from enviroment import Env
from centralities import get_centrality_node, centrality_heuristic
from modules.graph.Grafo import GrafoListaAdj
from modules.utils import read_Instances
from modules.utils.handle_labels import Bf_graph

CENTRALITIES={
    "Degree":{"func":nx.degree_centrality,"args":{},"reverse":True},
    "Closeness":{"func":nx.closeness_centrality,"args":{},"reverse":True},
    "Betweenness":{"func":nx.betweenness_centrality,"args":{},"reverse":True},
    "Eigenvector":{"func":nx.eigenvector_centrality,"args":{"max_iter":1000,"tol":1e-6},"reverse":True},
    "Katz Centrality":{"func":nx.katz_centrality,"args":{"alpha":0.005,"beta":1.0,"max_iter":1000,"tol":1e-6},"reverse":True},
    "PageRank":{"func":nx.pagerank,"args":{"alpha":0.85},"reverse":True},
    "Harmonic Centrality":{"func":nx.harmonic_centrality,"args":{},"reverse":True},
}

def build(path):
    nnodes,nedges,edges,_,_,_=read_Instances.load_instance(path)
    G=nx.Graph();G.add_nodes_from(range(1,nnodes+1));G.add_edges_from(edges)
    g=GrafoListaAdj();g.DefinirN(nnodes,VizinhancaDuplamenteLigada=True)
    for u,v in edges:g.AdicionarAresta(u,v)
    positions={node:i for i,node in enumerate(G.nodes())}
    original=max((abs(positions[u]-positions[v]) for u,v in G.edges()),default=0)
    maps={name:get_centrality_node(G,cfg) for name,cfg in CENTRALITIES.items()}
    return nnodes,nedges,G,g,original,maps

def eval_centrality(g,maps,name,budget):
    best=float("inf")
    # centrality_heuristic does iter_max constructions. Match DQN budget approximately:
    # DQN step internally performs 5 calls x 3 constructions = 15 constructions.
    constructions=budget*15
    cost,_,_,_=centrality_heuristic(
        graph=g,centrality_values=maps[name],cent_str=name,
        alpha=0.3,iter_max=constructions,centralities=CENTRALITIES)
    return int(cost)

def eval_random(g,maps,budget,seed):
    rng=random.Random(seed)
    best=float("inf")
    names=list(CENTRALITIES)
    for _ in range(budget):
        name=rng.choice(names)
        # Match one DQN step: 5 calls x 3 constructions = 15 samples.
        for _ in range(5):
            cost,_,_,_=centrality_heuristic(
                graph=g,centrality_values=maps[name],cent_str=name,
                alpha=0.3,iter_max=3,centralities=CENTRALITIES)
            best=min(best,cost)
    return int(best)

def eval_rcm(G,g):
    order=list(nx.utils.reverse_cuthill_mckee_ordering(G))
    labels={node:i+1 for i,node in enumerate(order)}
    return int(Bf_graph(g,labels))

def eval_dqn(g,maps,original,budget):
    names=list(CENTRALITIES)
    agent=Agent(learning_rate=.001,gamma=.9,epsilon=1.0,eps_min=.01,eps_dec=.995,
                n_movements=len(names),n_actions=len(names),n_states=4,deep=True)
    env=Env(original);state=env.get_initial_state();best=original
    for i in range(budget):
        action=agent.choose_action(state);name=names[action]
        info=env.step(graph=g,centrality_values=maps[name],cent_str=name,centralities=CENTRALITIES)
        ns=[info["n_steps"],info["gap"],info["reward"],info["bandwidth"]]
        agent.learn(state,action,info["reward"],ns,done=(i==budget-1))
        state=ns;best=min(best,info["bandwidth"])
    return int(best)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True);p.add_argument("--seeds",type=int,default=30)
    p.add_argument("--budget",type=int,default=30);p.add_argument("--output",required=True)
    p.add_argument("instances",nargs="+");a=p.parse_args()
    rows=[]
    for fname in a.instances:
        path=os.path.join(a.data_dir,fname)
        nnodes,nedges,G,g,orig,maps=build(path)
        rcm=eval_rcm(G,g)
        for seed in range(1,a.seeds+1):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            methods={}
            methods["DQN"]=eval_dqn(g,maps,orig,a.budget)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            methods["RandomCentrality"]=eval_random(g,maps,a.budget,seed)
            methods["RCM"]=rcm
            for name in CENTRALITIES:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                methods[name]=eval_centrality(g,maps,name,a.budget)
            for method,bw in methods.items():
                rows.append({"instance":fname,"seed":seed,"method":method,"nodes":nnodes,"edges":nedges,
                             "original_bw":orig,"best_bw":bw,
                             "reduction_pct":0.0 if orig==0 else 100*(orig-bw)/orig})
    fields=list(rows[0].keys())
    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(f"wrote {len(rows)} rows")

if __name__=="__main__":main()
