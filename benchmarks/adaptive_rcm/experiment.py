"""Sequential RCM local refinement: richer-state adaptive Double DQN vs discounted UCB."""
import sys,random,copy,time,csv,argparse,json
from collections import deque
from pathlib import Path
import numpy as np,networkx as nx,torch
from torch import nn
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'dynamic_rcm'))
from dynamic_rcm import Env as OldEnv, instance, get_real, load_mtx, SOURCE

class RichEnv(OldEnv):
 def reset(self):
  self.history=deque(maxlen=20);self.count=np.zeros(4);self.success=np.zeros(4);self.ewma=np.zeros(4);self.prev_accepted=0;self.prev_bw=0.;super().reset();self.prev_bw=float(self.value[0]);return self.state()
 def state(self):
  base=super().state().tolist();bw,c=self.value
  crit=[(u,v) for u,v in self.edges if abs(self.position[u]-self.position[v])==bw]
  endpoints=[v for e in crit for v in e]; deg=dict(self.g.degree())
  ds=np.array([deg[v] for v in endpoints],dtype=float) if endpoints else np.array([0.])
  recent=np.array(list(self.history),dtype=float) if self.history else np.zeros(1)
  frac=np.divide(self.success,self.count,out=np.zeros(4),where=self.count>0)
  extra=[len(set(endpoints))/max(1,self.n),float(ds.mean())/max(1,self.n-1),float(ds.max())/max(1,self.n-1),
         np.std(ds)/max(1,self.n-1),float(recent.mean()),float(self.prev_accepted),
         float(self.prev_bw-self.value[0])/max(1,self.initial[0]),
         *frac.tolist(),*self.ewma.tolist(),*(self.count/max(1,self.t)).tolist()]
  return np.asarray(base+extra,dtype=np.float32)
 def step(self,arm):
  old=self.value
  _,rew,done,info=super().step(arm)
  accepted=int(info['accepted']);self.history.append(accepted)
  self.count[arm]+=1;self.success[arm]+=accepted;self.ewma[arm]=.86*self.ewma[arm]+.14*rew
  self.prev_accepted=accepted;self.prev_bw=old[0]
  return self.state(),rew,done,info

class Net(nn.Module):
 def __init__(self,d):
  super().__init__();self.layers=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
 def forward(self,x):return self.layers(x)

class DQN:
 def __init__(self,seed,d):
  torch.manual_seed(seed);self.net=Net(d);self.target=copy.deepcopy(self.net);self.opt=torch.optim.Adam(self.net.parameters(),lr=.0005)
  self.replay=deque(maxlen=25000);self.rng=random.Random(seed);self.updates=0
 def act(self,s,eps=0.):
  if self.rng.random()<eps:return self.rng.randrange(4)
  with torch.no_grad():return int(self.net(torch.from_numpy(s).unsqueeze(0)).argmax().item())
 def learn(self):
  if len(self.replay)<64:return
  ex=self.rng.sample(self.replay,64)
  s=torch.from_numpy(np.stack([v[0] for v in ex]));ns=torch.from_numpy(np.stack([v[3] for v in ex]))
  a=torch.tensor([v[1] for v in ex],dtype=torch.long);r=torch.tensor([v[2] for v in ex],dtype=torch.float32);done=torch.tensor([v[4] for v in ex],dtype=torch.float32)
  q=self.net(s).gather(1,a[:,None]).squeeze(1)
  with torch.no_grad():target=r+.95*(1-done)*self.target(ns).gather(1,self.net(ns).argmax(1)[:,None]).squeeze(1)
  loss=nn.functional.smooth_l1_loss(q,target)
  self.opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),1.);self.opt.step();self.updates+=1
  if self.updates%150==0:self.target.load_state_dict(self.net.state_dict())

class DiscountedUCB:
 def __init__(self,decay=.95,c=.22):self.decay=decay;self.c=c;self.w=np.zeros(4);self.val=np.zeros(4)
 def act(self):
  for a in range(4):
   if self.w[a]<.001:return a
  return int(np.argmax(self.val/np.maximum(self.w,1e-9)+self.c*np.sqrt(np.log(1+sum(self.w))/np.maximum(self.w,1e-9))))
 def update(self,arm,reward):
  self.w*=self.decay;self.val*=self.decay;self.w[arm]+=1;self.val[arm]+=reward

def adaptive_epsilon(env,episode=0,training=False):
 base=max(.08,.85*(1-episode/90)**1.3) if training else .025
 return min(.50,base+.28*min(env.stalled/18,1))

def train(agent,graphs,steps):
 t0=time.perf_counter();by_ep=[]
 for i,(name,family,g) in enumerate(graphs):
  e=RichEnv(g,steps,100000+i*137);s=e.reset();reward_sum=0
  for t in range(steps):
   a=agent.act(s,adaptive_epsilon(e,i,True));ns,r,done,_=e.step(a)
   agent.replay.append((s,a,r,ns,done));reward_sum+=r
   if t%3==0:agent.learn()
   s=ns
  by_ep.append(dict(graph=name,reward=reward_sum,rcm=e.initial[0],final=e.value[0]))
 return time.perf_counter()-t0,by_ep

def evaluate(agent,graphs,reps,steps):
 rows=[];actions=[]
 policies=['RCM','Random_dynamic','Fixed_adjacent','Fixed_neighbor','DQN_greedy','DQN_adaptive','Discounted_UCB']
 for gi,(name,family,g) in enumerate(graphs):
  for rep in range(reps):
   seed=73400+gi*379+rep*211
   for policy in policies:
    t0=time.perf_counter();env=RichEnv(g,steps,seed);s=env.reset();rr=random.Random(seed+999);bandit=DiscountedUCB();hist=[]
    if policy!='RCM':
     for t in range(steps):
      if policy=='Random_dynamic':a=rr.randrange(4)
      elif policy=='Fixed_adjacent':a=3
      elif policy=='Fixed_neighbor':a=2
      elif policy=='DQN_greedy':a=agent.act(s,0)
      elif policy=='DQN_adaptive':
       if rr.random()<adaptive_epsilon(env):a=rr.randrange(4)
       else:a=agent.act(s,0)
      else:a=bandit.act()
      s,r,done,info=env.step(a)
      if policy=='Discounted_UCB':bandit.update(a,r)
      hist.append(a)
    elapsed=time.perf_counter()-t0
    rows.append(dict(graph=name,family=family,rep=rep,method=policy,n=len(g),edges=g.number_of_edges(),rcm=env.initial[0],bw=env.value[0],gain=env.initial[0]-env.value[0],gain_pct=100*(env.initial[0]-env.value[0])/max(1,env.initial[0]),seconds=elapsed,accepted=env.improvements,steps=0 if policy=='RCM' else steps))
    for a in range(4):actions.append(dict(graph=name,rep=rep,method=policy,action=a,count=hist.count(a)))
  print('EVAL',name,flush=True)
 return rows,actions

def output_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=int,default=65);p.add_argument('--test',type=int,default=20);p.add_argument('--steps',type=int,default=100);p.add_argument('--reps',type=int,default=3);args=p.parse_args()
 torch.set_num_threads(1)
 kinds=['ER','BA','WS','grid','geometric']
 tr=[(f'train_{i}',kinds[i%5],instance('train',kinds[i%5],36+(i*29)%104,5000+i*131)) for i in range(args.train)]
 te=[(f'test_{i}',kinds[i%5],instance('test',kinds[i%5],36+(i*37)%104,8000+i*157)) for i in range(args.test)]
 real=[(k,'real',g) for k,g in get_real().items()]
 real.append(('bcsstk01','real',load_mtx(SOURCE/'suitesparse_benchmark/bcsstk01.mtx')))
 state_dim=len(RichEnv(tr[0][2],args.steps,1).state())
 agent=DQN(47,state_dim)
 duration,train_rows=train(agent,tr,args.steps)
 rows,actions=evaluate(agent,te+real,args.reps,args.steps)
 output_csv(ROOT/'results.csv',rows);output_csv(ROOT/'actions.csv',actions);output_csv(ROOT/'train.csv',train_rows)
 summary=[];stats={}
 for split in ('synthetic','real'):
  d=[r for r in rows if (r['family']=='real')==(split=='real')]
  names=sorted({r['graph'] for r in d});methods=sorted({r['method'] for r in d})
  bymethod={m:np.asarray([np.mean([r['bw'] for r in d if r['graph']==name and r['method']==m]) for name in names]) for m in methods}
  for m in methods:
   z=[r for r in d if r['method']==m]
   summary.append(dict(split=split,method=m,graphs=len(names),mean_bw=float(np.mean([r['bw'] for r in z])),mean_gain_pct=float(np.mean([r['gain_pct'] for r in z])),mean_seconds=float(np.mean([r['seconds'] for r in z]))))
  stats[split]={}
  for m in ['Random_dynamic','Fixed_adjacent','Fixed_neighbor','RCM','DQN_greedy','Discounted_UCB']:
   diff=bymethod['DQN_adaptive']-bymethod[m]
   stats[split][m]=dict(delta=float(np.mean(diff)),wins=int(sum(diff<0)),losses=int(sum(diff>0)),ties=int(sum(diff==0)),p=float(wilcoxon(diff).pvalue) if np.any(diff!=0) else None)
 output_csv(ROOT/'summary.csv',summary)
 (ROOT/'stats.json').write_text(json.dumps(dict(train_graphs=args.train,test_graphs=args.test,real_graphs=len(real),reps=args.reps,steps=args.steps,state_dimensions=state_dim,train_seconds=duration,updates=agent.updates,comparisons=stats),indent=2))
 torch.save(agent.net.state_dict(),ROOT/'weights.pth')
 print('SUMMARY',json.dumps(summary,indent=2),flush=True)
 print('STATS',json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
