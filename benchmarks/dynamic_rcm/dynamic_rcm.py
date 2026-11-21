"""Dynamic operator-selection Double DQN from RCM, with matched step budgets.
Train: python dynamic_rcm.py --train 130 --test 35 --reps 6 --steps 150
"""
import argparse, copy, csv, json, random, sys, time
from collections import deque
from pathlib import Path
import numpy as np
import networkx as nx
import torch
from torch import nn
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parent
from operators import proposal, instance

def get_real():
    raise RuntimeError('Use benchmark.py with real files from the repository; packaged real-data fixtures are not available.')

def load_mtx(path):
    raise RuntimeError('Use benchmark.graph_from_mtx for Matrix Market files.')

SOURCE=ROOT

def band_stats(edges,pos):
    if not edges: return (0,0)
    gaps=[abs(pos[u]-pos[v]) for u,v in edges]
    bw=max(gaps)
    return bw,sum(x==bw for x in gaps)

class Env:
    def __init__(self,g,steps,seed):
        self.g=g;self.edges=list(g.edges());self.n=len(g);self.steps=steps;self.seed=seed
        self.base=list(nx.utils.reverse_cuthill_mckee_ordering(g)); self.initial=band_stats(self.edges,{v:i for i,v in enumerate(self.base)})
        self.reset()
    def reset(self):
        self.order=list(self.base);self.position={v:i for i,v in enumerate(self.order)}
        self.value=self.initial;self.t=0;self.last_gain=0.;self.improvements=0;self.stalled=0;self.last_arm=-1
        return self.state()
    def state(self):
        bw,crit=self.value
        vals=[np.log1p(self.n)/7, np.log1p(len(self.edges))/9, self.t/self.steps,
              bw/max(1,self.n-1), bw/max(1,self.initial[0]),
              crit/max(1,len(self.edges)), min(1,self.stalled/35),
              self.last_gain, self.improvements/max(1,self.t), self.last_arm/3 if self.last_arm>=0 else -1]
        return np.array(vals,dtype=np.float32)
    def step(self,arm):
        old=self.value
        # fresh action/step-specific random sequence, shared across evaluation methods
        rng=random.Random(self.seed+self.t*10007+int(arm)*1000003)
        i,j=proposal(int(arm),self.order,self.edges,rng) if self.n>1 and self.edges else (0,0)
        if i!=j:
            a,b=self.order[i],self.order[j]
            self.order[i],self.order[j]=b,a;self.position[a]=j;self.position[b]=i
            new=band_stats(self.edges,self.position)
            if new<old:self.value=new;accepted=True
            else:
                self.order[i],self.order[j]=a,b;self.position[a]=i;self.position[b]=j;accepted=False
        else:accepted=False
        gain=old[0]-self.value[0];crit_gain=old[1]-self.value[1]
        # positive only for accepted improving moves; shaping for critical edges, not a substitute for primary metric
        reward=3*gain+0.4*max(0,crit_gain)/max(1,old[1])
        self.t+=1;self.last_gain=float(gain>0);self.improvements+=int(accepted);self.stalled=0 if accepted else self.stalled+1;self.last_arm=int(arm)
        done=self.t>=self.steps
        return self.state(),float(reward),done,dict(bandwidth=self.value[0],critical=self.value[1],accepted=accepted)

class Net(nn.Module):
    def __init__(self):
        super().__init__();self.m=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,x):return self.m(x)

class DQN:
    def __init__(self,seed):
        torch.manual_seed(seed);self.online=Net();self.target=copy.deepcopy(self.online)
        self.opt=torch.optim.Adam(self.online.parameters(),lr=0.0006)
        self.replay=deque(maxlen=15000);self.rng=random.Random(seed);self.learn_steps=0
    def act(self,s,epsilon=0):
        if self.rng.random()<epsilon:return self.rng.randrange(4)
        with torch.no_grad():return int(self.online(torch.tensor(s).reshape(1,-1)).argmax().item())
    def update(self,batch=64):
        if len(self.replay)<batch:return
        examples=self.rng.sample(self.replay,batch)
        s=torch.tensor(np.stack([x[0] for x in examples]));a=torch.tensor([x[1] for x in examples]).long()
        r=torch.tensor([x[2] for x in examples]).float();ns=torch.tensor(np.stack([x[3] for x in examples])); done=torch.tensor([x[4] for x in examples]).float()
        q=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            best=self.online(ns).argmax(dim=1)
            target=r+0.95*(1-done)*self.target(ns).gather(1,best[:,None]).squeeze(1)
        loss=nn.functional.smooth_l1_loss(q,target)
        self.opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.online.parameters(),1.)
        self.opt.step();self.learn_steps+=1
        if self.learn_steps%120==0:self.target.load_state_dict(self.online.state_dict())

def train(agent,count,steps,seed):
    rng=np.random.default_rng(seed)
    start=time.perf_counter()
    for i in range(count):
        kind=['ER','BA','WS','grid','geometric'][i%5]
        g=instance('train',kind,38+(i*37)%115,seed+177*i)
        e=Env(g,steps,seed+300011+i*199)
        s=e.reset();epsilon=max(0.08,0.95*(1-i/max(1,count))**1.35)
        for t in range(steps):
            arm=agent.act(s,epsilon);ns,reward,done,_=e.step(arm)
            agent.replay.append((s,arm,reward,ns,done))
            if t%4==0:agent.update()
            s=ns
    return time.perf_counter()-start

def evaluate(agent,graphs,reps,steps):
    records=[]
    methods=['RCM','Random_dynamic','Fixed_random_swap','Fixed_midpoint','Fixed_neighbor','Fixed_adjacent','DQN_dynamic']
    for ix,(name,fam,g) in enumerate(graphs):
        for rep in range(reps):
            seed=2026000+ix*4111+rep*103
            for method in methods:
                env=Env(g,steps,seed);s=env.reset();t0=time.perf_counter()
                if method!='RCM':
                    rr=random.Random(seed+999)
                    for t in range(steps):
                        if method=='DQN_dynamic':arm=agent.act(s,0)
                        elif method=='Random_dynamic':arm=rr.randrange(4)
                        else:arm={'Fixed_random_swap':0,'Fixed_midpoint':1,'Fixed_neighbor':2,'Fixed_adjacent':3}[method]
                        s,r,done,info=env.step(arm)
                secs=time.perf_counter()-t0
                records.append(dict(graph=name,family=fam,n=len(g),edges=g.number_of_edges(),rep=rep,method=method,
                    rcm=env.initial[0],bw=env.value[0],improvement=env.initial[0]-env.value[0],rel_improvement=(env.initial[0]-env.value[0])/max(1,env.initial[0]),
                    seconds=secs,accepted=env.improvements,steps=(steps if method!='RCM' else 0)))
        print('EVAL',name, 'RCM',records[-1]['rcm'],flush=True)
    return records

def summarize(rows,tag):
    d=[x for x in rows if x['family'].startswith(tag)]
    methods=sorted(set(x['method'] for x in d))
    out=[]
    for m in methods:
        z=[x for x in d if x['method']==m]
        out.append(dict(split=tag,method=m,graphs=len(set(x['graph'] for x in z)),mean_bw=np.mean([x['bw'] for x in z]),mean_improvement=np.mean([x['improvement'] for x in z]),mean_rel_improvement=np.mean([x['rel_improvement'] for x in z]),mean_seconds=np.mean([x['seconds'] for x in z])))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',type=int,default=100);p.add_argument('--test',type=int,default=30);p.add_argument('--steps',type=int,default=150);p.add_argument('--reps',type=int,default=5);a=p.parse_args()
    torch.set_num_threads(1)
    agent=DQN(47);t_train=train(agent,a.train,a.steps,170000)
    tests=[(f'{kind}_{i}',f'synthetic_{kind}',instance('test',kind,38+(i*31)%115,899000+i*113)) for i in range(a.test) for kind in [['ER','BA','WS','grid','geometric'][i%5]]]
    real=[(k,'real',g) for k,g in get_real().items()]
    real.append(('bcsstk01','real',load_mtx(SOURCE/'suitesparse_benchmark/bcsstk01.mtx')))
    rows=evaluate(agent,tests+real,a.reps,a.steps)
    for file,items in [('results.csv',rows),('summary.csv',summarize(rows,'synthetic')+summarize(rows,'real'))]:
        with (ROOT/file).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    out={'seed':47,'training_seconds':t_train,'train_graphs':a.train,'test_synthetic':a.test,'real_graphs':len(real),'repetitions':a.reps,'steps':a.steps,'updates':agent.learn_steps}
    for tag in ['synthetic','real']:
        z=[r for r in rows if r['family'].startswith(tag)];names=sorted(set(x['graph'] for x in z)); means={m:np.array([np.mean([x['bw'] for x in z if x['graph']==name and x['method']==m]) for name in names]) for m in set(x['method'] for x in z)}
        out[tag]={}
        for m in ['RCM','Random_dynamic','Fixed_random_swap','Fixed_midpoint','Fixed_neighbor','Fixed_adjacent']:
            delta=means['DQN_dynamic']-means[m]
            out[tag][m]={'mean_difference':float(delta.mean()),'wins':int((delta<0).sum()),'losses':int((delta>0).sum()),'ties':int((delta==0).sum()),'p_wilcoxon':float(wilcoxon(delta).pvalue) if np.any(delta!=0) else None}
    (ROOT/'diagnostics.json').write_text(json.dumps(out,indent=2));torch.save(agent.online.state_dict(),ROOT/'dqn_weights.pth')
    print('DONE',json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
