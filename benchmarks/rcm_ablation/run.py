"""RCM dynamic-operator study: 10D baseline, 29D enriched, 29D ablations, adaptive exploration, Discounted-UCB.
Training uses ONLY synthetic graphs. Real matrices are fully held out; family clustering reported.
"""
import sys, time, random, copy, csv, json, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'adaptive_rcm'))
from experiment import RichEnv, DQN, DiscountedUCB, adaptive_epsilon
sys.path.insert(0,str(ROOT.parent/'dynamic_rcm'))
from dynamic_rcm import Env, instance, get_real, load_mtx, SOURCE

def graphs_real():
    x=get_real(); x['bcsstk01']=load_mtx(SOURCE/'suitesparse_benchmark/bcsstk01.mtx')
    families={'bcsstk01':'structural','dwt59':'structural_dwt','bfwa62':'electromagnetic_bfw','bfwb62':'electromagnetic_bfw','lns_131':'fluid_dynamics'}
    return [(k,families[k],v) for k,v in x.items()]

def ablate(state, mode):
    z=np.array(state,copy=True)
    if mode=='drop_history':z[14:]=0.0
    if mode=='drop_critical':z[10:14]=0.0
    return z

def train(seed, mode, count, steps):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    dim=10 if mode=='original' else 29
    agent=DQN(seed,dim)
    kinds=['ER','BA','WS','grid','geometric']
    start=time.perf_counter()
    train_rows=[]
    for i in range(count):
        k=kinds[i%len(kinds)];g=instance('train',k,36+(i*29)%94,10103+i*173)
        e=(Env if mode=='original' else RichEnv)(g,steps,seed*900000+i*173);s=ablate(e.reset(),mode) if mode!='original' else e.reset()
        for t in range(steps):
            eps=max(.08,.90*(1-i/count)**1.3)
            a=agent.act(s,eps)
            ns,r,done,_=e.step(a)
            if mode!='original':ns=ablate(ns,mode)
            agent.replay.append((s,a,r,ns,done))
            if t%3==0:agent.learn()
            s=ns
        train_rows.append({'seed':seed,'model':mode,'graph':i,'family':k,'rcm':e.initial[0],'final':e.value[0]})
    return agent,time.perf_counter()-start,train_rows

def evaluate(agents,seed,graphs,reps,steps):
    res=[];act=[]
    for gidx,(name,family,g) in enumerate(graphs):
        for rep in range(reps):
            trialseed=910000+gidx*1009+rep*317
            methods=['RCM','Random','Fixed_adjacent','DQN_original','DQN_rich','DQN_no_history','DQN_no_critical','DQN_rich_adaptive','Discounted_UCB']
            for method in methods:
                e=RichEnv(g,steps,trialseed);s=e.reset(); rr=random.Random(trialseed+13)
                bandit=DiscountedUCB();counts=Counter();start=time.perf_counter()
                for t in range(0 if method=='RCM' else steps):
                    if method=='Random':a=rr.randrange(4)
                    elif method=='Fixed_adjacent':a=3
                    elif method=='Discounted_UCB':a=bandit.act()
                    else:
                        mode={'DQN_original':'original','DQN_rich':'rich','DQN_no_history':'drop_history','DQN_no_critical':'drop_critical','DQN_rich_adaptive':'rich'}[method]
                        z=e.state()[:10] if mode=='original' else ablate(e.state(),mode)
                        if method=='DQN_rich_adaptive' and rr.random()<adaptive_epsilon(e): a=rr.randrange(4)
                        else:a=agents[mode].act(z,0.)
                    _,rew,_,info=e.step(a)
                    if method=='Discounted_UCB':bandit.update(a,rew)
                    counts[a]+=1
                secs=time.perf_counter()-start
                res.append({'seed':seed,'graph':name,'family':family,'rep':rep,'method':method,'n':len(g),'m':g.number_of_edges(),'rcm':e.initial[0],'bw':e.value[0],'gain':e.initial[0]-e.value[0],'seconds_refine':secs,'steps':0 if method=='RCM' else steps})
                for a in range(4):act.append({'seed':seed,'graph':name,'rep':rep,'method':method,'arm':a,'count':counts[a]})
        print('EVAL',seed,name,flush=True)
    return res,act

def write_csv(path,data):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=int,default=55);ap.add_argument('--steps',type=int,default=100);ap.add_argument('--reps',type=int,default=3);ap.add_argument('--seeds',type=int,default=3);p=ap.parse_args()
    torch.set_num_threads(1)
    real=graphs_real();all_rows=[];all_actions=[];all_training=[];times=[]
    for i in range(p.seeds):
        seed=47+31*i
        agents={}
        for mode in ['original','rich','drop_history','drop_critical']:
            ag,sec,train_rows=train(seed,mode,p.train,p.steps); agents[mode]=ag
            times.append({'seed':seed,'model':mode,'train_seconds':sec,'updates':ag.updates,'train_graphs':p.train,'steps':p.steps})
            all_training+=train_rows
            print('TRAIN',seed,mode,round(sec,2),flush=True)
        rows,actions=evaluate(agents,seed,real,p.reps,p.steps)
        all_rows+=rows; all_actions+=actions
    write_csv(ROOT/'results.csv',all_rows);write_csv(ROOT/'actions.csv',all_actions);write_csv(ROOT/'training.csv',times);write_csv(ROOT/'training_curves.csv',all_training)
    group=[]
    import pandas as pd
    df=pd.DataFrame(all_rows)
    for (family,method),v in df.groupby(['family','method']):
        group.append({'family':family,'method':method,'matrices':v.graph.nunique(),'mean_bw':v.bw.mean(),'mean_gain':v.gain.mean(),'mean_refine_ms':1000*v.seconds_refine.mean()})
    write_csv(ROOT/'family_summary.csv',group)
    allmeans=df.groupby('method').agg(mean_bw=('bw','mean'),mean_gain=('gain','mean'),mean_refine_ms=('seconds_refine',lambda x:1000*x.mean())).reset_index()
    allmeans.to_csv(ROOT/'summary.csv',index=False)
    matrixmeans=df.groupby(['graph','family','method']).bw.mean().unstack('method')
    comparisons={}
    for a in ['DQN_original','DQN_rich','DQN_no_history','DQN_no_critical','DQN_rich_adaptive','Discounted_UCB']:
        comparisons[a]={}
        for b in ['RCM','Random','Fixed_adjacent','DQN_original','DQN_rich']:
            if a==b:continue
            d=matrixmeans[a]-matrixmeans[b]
            comparisons[a][b]={'mean_delta_bw':float(d.mean()),'wins':int((d<-1e-9).sum()),'losses':int((d>1e-9).sum()),'ties':int((abs(d)<=1e-9).sum()),'p_wilcoxon_exploratory':float(wilcoxon(d).pvalue) if (abs(d)>1e-9).any() else None}
    (ROOT/'diagnostics.json').write_text(json.dumps({'train_only':'synthetic','real_test_graphs':len(real),'real_test_families':len(set(x[1] for x in real)),'train_graphs_per_seed':p.train,'training_seeds':p.seeds,'evaluation_reps':p.reps,'steps':p.steps,'comps_by_matrix':comparisons,'family_names':{x[0]:x[1] for x in real},'warning':'Family labels inferred from names; few real graphs; no significance claimed; model training is extra computation.'},indent=2))
    print('SUMMARY\n',allmeans.to_string(index=False),flush=True)
    print('COMP\n',json.dumps(comparisons,indent=2),flush=True)
if __name__=='__main__':main()
