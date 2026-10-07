import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ROOT=os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)

from modules.utils import read_Instances


def canonical_bandwidth(G, order):
    pos={node:i for i,node in enumerate(order)}
    return max((abs(pos[u]-pos[v]) for u,v in G.edges()),default=0)


def rcm_order(G):
    return list(nx.utils.reverse_cuthill_mckee_ordering(G))


def normalized_adjacency(G, nodes, device):
    n=len(nodes)
    idx={v:i for i,v in enumerate(nodes)}
    A=torch.zeros((n,n),dtype=torch.float32,device=device)
    for u,v in G.edges():
        i,j=idx[u],idx[v]
        A[i,j]=1.0
        A[j,i]=1.0
    A=A+torch.eye(n,device=device)
    deg=A.sum(dim=1)
    d=deg.clamp(min=1e-8).pow(-0.5)
    return d[:,None]*A*d[None,:]


def node_features(G, nodes, order, initial_order, device):
    n=len(nodes)
    pos={v:i for i,v in enumerate(order)}
    init={v:i for i,v in enumerate(initial_order)}
    maxdeg=max((G.degree(v) for v in nodes),default=1)
    bw=max(1,canonical_bandwidth(G,order))
    feats=[]
    for v in nodes:
        local=max((abs(pos[v]-pos[u]) for u in G.neighbors(v)),default=0)
        feats.append([
            G.degree(v)/max(1,maxdeg),
            pos[v]/max(1,n-1),
            init[v]/max(1,n-1),
            local/bw,
        ])
    return torch.tensor(feats,dtype=torch.float32,device=device)


class GCNEncoder(nn.Module):
    def __init__(self,in_dim=4,hidden=64,out_dim=64):
        super().__init__()
        self.lin1=nn.Linear(in_dim,hidden)
        self.lin2=nn.Linear(hidden,out_dim)

    def forward(self,x,A):
        h=torch.relu(self.lin1(A@x))
        h=torch.relu(self.lin2(A@h))
        return h


class ActorCritic(nn.Module):
    def __init__(self,embed_dim=64):
        super().__init__()
        self.encoder=GCNEncoder(4,64,embed_dim)
        self.actor=nn.Sequential(
            nn.Linear(embed_dim*2,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
        self.critic=nn.Sequential(
            nn.Linear(embed_dim,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )

    def forward(self,x,A,order_idx):
        h=self.encoder(x,A)
        ordered=h[order_idx]
        if ordered.shape[0] < 2:
            logits=torch.zeros(1,device=h.device)
        else:
            pair=torch.cat([ordered[:-1],ordered[1:]],dim=1)
            logits=self.actor(pair).squeeze(-1)
        value=self.critic(h.mean(dim=0)).squeeze(-1)
        return logits,value


@dataclass
class Transition:
    x: torch.Tensor
    order_idx: torch.Tensor
    action: int
    logp: float
    reward: float
    done: bool
    value: float


class BandwidthEnv:
    def __init__(self,G,device):
        self.G=G
        self.nodes=list(G.nodes())
        self.device=device
        self.A=normalized_adjacency(G,self.nodes,device)
        self.initial_order=rcm_order(G)
        self.reset()

    def reset(self):
        self.order=list(self.initial_order)
        self.initial_bw=canonical_bandwidth(self.G,self.order)
        self.current_bw=self.initial_bw
        self.best_bw=self.current_bw
        self.best_order=list(self.order)
        return self.state()

    def state(self):
        x=node_features(self.G,self.nodes,self.order,self.initial_order,self.device)
        node_to_idx={v:i for i,v in enumerate(self.nodes)}
        order_idx=torch.tensor([node_to_idx[v] for v in self.order],dtype=torch.long,device=self.device)
        return x,order_idx

    def step(self,action):
        if len(self.order)<2:
            return self.state(),0.0,True
        i=int(max(0,min(action,len(self.order)-2)))
        old=self.current_bw
        self.order[i],self.order[i+1]=self.order[i+1],self.order[i]
        self.current_bw=canonical_bandwidth(self.G,self.order)
        reward=(old-self.current_bw)/max(1,self.initial_bw)
        if self.current_bw<self.best_bw:
            self.best_bw=self.current_bw
            self.best_order=list(self.order)
        return self.state(),float(reward),False


class PPOTrainer:
    def __init__(self,G,lr=3e-4,gamma=.99,gae_lambda=.95,clip=.2,epochs=4,device=None):
        self.device=device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.env=BandwidthEnv(G,self.device)
        self.model=ActorCritic().to(self.device)
        self.opt=torch.optim.Adam(self.model.parameters(),lr=lr)
        self.gamma=gamma
        self.gae_lambda=gae_lambda
        self.clip=clip
        self.epochs=epochs

    def collect(self,steps):
        transitions=[]
        x,order_idx=self.env.reset()
        for t in range(steps):
            with torch.no_grad():
                logits,value=self.model(x,self.env.A,order_idx)
                dist=Categorical(logits=logits)
                action=dist.sample()
                logp=dist.log_prob(action)
            (nx_state,no_idx),reward,_=self.env.step(action.item())
            done=(t==steps-1)
            transitions.append(Transition(
                x=x.detach(),order_idx=order_idx.detach(),action=int(action.item()),
                logp=float(logp.item()),reward=float(reward),done=done,value=float(value.item())
            ))
            x,order_idx=nx_state,no_idx
        return transitions,self.env.best_bw,self.env.best_order

    def _advantages(self,tr):
        rewards=[t.reward for t in tr]
        values=[t.value for t in tr]+[0.0]
        adv=[0.0]*len(tr)
        gae=0.0
        for i in reversed(range(len(tr))):
            delta=rewards[i]+self.gamma*values[i+1]-values[i]
            gae=delta+self.gamma*self.gae_lambda*gae
            adv[i]=gae
        returns=[adv[i]+values[i] for i in range(len(tr))]
        a=torch.tensor(adv,dtype=torch.float32,device=self.device)
        if len(a)>1 and float(a.std())>1e-8:
            a=(a-a.mean())/(a.std()+1e-8)
        return a,torch.tensor(returns,dtype=torch.float32,device=self.device)

    def update(self,tr):
        adv,returns=self._advantages(tr)
        old_logp=torch.tensor([t.logp for t in tr],dtype=torch.float32,device=self.device)
        for _ in range(self.epochs):
            losses=[]
            for i,t in enumerate(tr):
                logits,value=self.model(t.x,self.env.A,t.order_idx)
                dist=Categorical(logits=logits)
                action=torch.tensor(t.action,device=self.device)
                logp=dist.log_prob(action)
                ratio=torch.exp(logp-old_logp[i])
                s1=ratio*adv[i]
                s2=torch.clamp(ratio,1-self.clip,1+self.clip)*adv[i]
                actor_loss=-torch.min(s1,s2)
                critic_loss=0.5*(value-returns[i]).pow(2)
                entropy=dist.entropy()
                losses.append(actor_loss+critic_loss-0.01*entropy)
            loss=torch.stack(losses).mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
            self.opt.step()

    def train(self,episodes=20,steps=30,seed=123):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        global_best=self.env.initial_bw
        global_order=list(self.env.initial_order)
        history=[]
        for ep in range(episodes):
            tr,bw,order=self.collect(steps)
            self.update(tr)
            if bw<global_best:
                global_best=bw
                global_order=list(order)
            history.append({"episode":ep+1,"best_bw":int(bw),"global_best_bw":int(global_best)})
        return {
            "rcm_bw":int(self.env.initial_bw),
            "best_bw":int(global_best),
            "best_order":global_order,
            "history":history,
        }


def load_graph(path):
    nnodes,_,edges,_,_,_=read_Instances.load_instance(path)
    G=nx.Graph()
    G.add_nodes_from(range(1,nnodes+1))
    G.add_edges_from(edges)
    return G


def main():
    p=argparse.ArgumentParser()
    p.add_argument("instance")
    p.add_argument("--episodes",type=int,default=20)
    p.add_argument("--steps",type=int,default=30)
    p.add_argument("--seed",type=int,default=123)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    G=load_graph(a.instance)
    trainer=PPOTrainer(G)
    result=trainer.train(episodes=a.episodes,steps=a.steps,seed=a.seed)
    result.update({
        "instance":os.path.basename(a.instance),
        "nodes":G.number_of_nodes(),
        "edges":G.number_of_edges(),
    })
    print(json.dumps(result,indent=2))
    if a.output:
        with open(a.output,"w") as f: json.dump(result,f,indent=2)


if __name__=="__main__":
    main()
