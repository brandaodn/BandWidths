"""Audited benchmark for BandWidths RCM local refinements.
Run with --repo-mtx path/to/BandWidths to discover all .mtx under repository;
without --repo-mtx uses the five prepackaged previously retrieved real graphs.
For large instances use --max-n and --max-edges, which are logged as exclusions.
Training of DQN remains synthetic; all real graphs are strictly test-only.
"""
import sys, argparse, time, json, csv, re, hashlib, random
from pathlib import Path
import numpy as np
import networkx as nx
import torch
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path[:0]=[str(BASE/'rcm_ablation'),str(BASE/'adaptive_rcm'),str(BASE/'dynamic_rcm'),str(BASE/'rcm_refinement')]
from run import train as train_ablation, ablate
from experiment import RichEnv, DiscountedUCB, adaptive_epsilon
from dynamic_rcm import Env as OriginalEnv, get_real, load_mtx, SOURCE

METHODS=['RCM','Random','Fixed_adjacent','Fixed_neighbor','DQN_original','DQN_rich','DQN_no_history','DQN_no_critical','DQN_rich_adaptive','Discounted_UCB']

def graph_from_mtx(path):
    """Supports standard Matrix Market coordinate (incl. pattern/symmetric) and legacy no-header coordinate.
    Returns simple *structural* graph, discarding numeric values and diagonal.
    Rectangular matrices excluded because symmetric permutation is the target.
    """
    with open(path,'r',errors='replace') as f:
        first=f.readline().strip()
        if first.startswith('%%MatrixMarket'):
            terms=first.lower().split()
            if len(terms)<5 or terms[1]!='matrix' or terms[2]!='coordinate':
                raise ValueError('unsupported MatrixMarket storage (not coordinate)')
            sym=terms[4]
            if sym not in ('general','symmetric','hermitian','skew-symmetric'):
                raise ValueError('unsupported MatrixMarket symmetry '+sym)
            for line in f:
                if line.strip() and not line.startswith('%'):
                    nrow,ncol,entries=map(int,line.split()[:3]);break
            else:raise ValueError('missing dimensions')
        else:
            while not first or first.startswith('%'):
                first=f.readline().strip()
                if not first:raise ValueError('missing dimensions')
            dims=list(map(int,first.split()[:3]))
            if len(dims)==3:nrow,ncol,entries=dims
            elif len(dims)==2:nrow,entries=dims;ncol=nrow
            else:raise ValueError('invalid legacy header')
        if nrow!=ncol: raise ValueError('rectangular matrix')
        if nrow<2:raise ValueError('n<2')
        g=nx.Graph();g.add_nodes_from(range(1,nrow+1));seen=0
        for line in f:
            if not line.strip() or line.startswith('%'):continue
            p=line.split()
            if len(p)<2:continue
            u,v=int(p[0]),int(p[1]);seen+=1
            if not (1<=u<=nrow and 1<=v<=nrow):raise ValueError('out-of-bounds edge')
            if u!=v:g.add_edge(u,v)
        if seen==0:raise ValueError('no coordinate entries')
        return g

def signature(g):
    h=hashlib.sha256();h.update(str(len(g)).encode())
    for u,v in sorted(tuple(sorted(e)) for e in g.edges()):h.update(f'{u},{v};'.encode())
    return h.hexdigest()

def family_of(path):
    s=path.stem.lower(); parent=path.parent.name.lower()
    # Structural prefix from SuiteSparse style filenames, rather than path's old 'trash' folder.
    for p in ['bcsstk','bcsstm','bcspwr','bfwa','bfwb','dwt','lshp','lnsp','lns','bp','gre','impcol','can','shl','orsirr']:
        if s.startswith(p):return p
    if parent not in ('trash','artigo_datas','matrix','old','big','newdata'):
        return parent
    return re.sub(r'[_-]?\d.*$','',s) or 'other'

def discover(root,max_n,max_edges):
    seen={};accepted=[];audit=[]
    files=sorted(root.rglob('*.mtx'))
    for p in files:
        reason=None;n=m=0; sig=None
        try:
            g=graph_from_mtx(p);n=len(g);m=g.number_of_edges();sig=signature(g)
            if sig in seen:reason='duplicate_topology_exact'
            elif n>max_n or m>max_edges:reason='size_limit'
            elif m==0:reason='edgeless'
            else:
                key=f'{p.stem}__{len(accepted)}'
                accepted.append((key,family_of(p),g));seen[sig]=str(p)
        except Exception as ex:reason=str(ex).split('\n')[0][:150]
        audit.append(dict(path=str(p),family=family_of(p),n=n,edges=m,signature=sig or '',status='included' if reason is None else 'excluded',reason=reason or '',duplicate_of=seen.get(sig,'') if reason=='duplicate_topology_exact' else ''))
    return accepted,audit

def bundled():
    g=get_real();g['bcsstk01']=load_mtx(SOURCE/'suitesparse_benchmark'/'bcsstk01.mtx')
    fam={'bcsstk01':'bcsstk','bfwa62':'bfwa','bfwb62':'bfwb','dwt59':'dwt','lns_131':'lns'}
    return [(k,fam[k],v) for k,v in sorted(g.items())]

def verify_env(g):
    e=RichEnv(g,20,123);s=e.reset();assert s.shape==(29,) and np.isfinite(s).all()
    assert len(e.order)==len(g) and len(set(e.order))==len(g)
    start=e.initial
    for a in range(4):
        prev=e.value
        s,r,d,info=e.step(a)
        assert e.value<=prev and e.initial==start
        assert len(set(e.order))==len(g) and set(e.order)==set(g)
        assert np.isfinite(s).all() and r>=0
    return start[0]

def evaluate(agents,graphs,seeds,reps,steps,trace_rows=None):
    out=[];actions=[]
    for gidx,(name,family,g) in enumerate(graphs):
        assert verify_env(g)>=0
        t_rcm=time.perf_counter()
        _=list(nx.utils.reverse_cuthill_mckee_ordering(g))
        rcm_sec=time.perf_counter()-t_rcm
        for rep in range(reps):
            trialseed=102049+gidx*2211+rep*371
            for seed,ag in agents.items():
                for method in METHODS:
                    env=RichEnv(g,steps,trialseed);s=env.reset();rng=random.Random(trialseed+55)
                    bandit=DiscountedUCB();num=[0]*4; t0=time.perf_counter()
                    checkpoints=set([0,1,steps]+[max(1,round(steps*j/20)) for j in range(1,21)])
                    if trace_rows is not None:
                        trace_rows.append(dict(graph=name,family=family,seed=seed,rep=rep,method=method,steps_budget=steps,step=0,bw=env.value[0],elapsed_refinement_sec=0.0,rcm=env.initial[0]))
                    for t in range(0 if method=='RCM' else steps):
                        if method=='Random':a=rng.randrange(4)
                        elif method=='Fixed_adjacent':a=3
                        elif method=='Fixed_neighbor':a=2
                        elif method=='Discounted_UCB':a=bandit.act()
                        else:
                            mode={'DQN_original':'original','DQN_rich':'rich','DQN_no_history':'drop_history','DQN_no_critical':'drop_critical','DQN_rich_adaptive':'rich'}[method]
                            z=env.state()[:10] if mode=='original' else ablate(env.state(),mode)
                            if method=='DQN_rich_adaptive' and rng.random()<adaptive_epsilon(env):a=rng.randrange(4)
                            else:a=ag[mode].act(z,0.)
                        s,reward,done,info=env.step(a)
                        if method=='Discounted_UCB':bandit.update(a,reward)
                        num[a]+=1
                        if trace_rows is not None and (t+1) in checkpoints:
                            trace_rows.append(dict(graph=name,family=family,seed=seed,rep=rep,method=method,steps_budget=steps,step=t+1,bw=env.value[0],elapsed_refinement_sec=time.perf_counter()-t0,rcm=env.initial[0]))
                    secs=time.perf_counter()-t0
                    out.append(dict(graph=name,family=family,seed=seed,rep=rep,method=method,n=len(g),m=g.number_of_edges(),rcm=env.initial[0],bw=env.value[0],accepted=env.improvements,steps=0 if method=='RCM' else steps,refinement_sec=secs,rcm_sec=rcm_sec,total_inference_sec=secs+rcm_sec))
                    for a,v in enumerate(num):actions.append(dict(graph=name,seed=seed,rep=rep,method=method,arm=a,count=v))
        print('TESTED',name,family,len(g),g.number_of_edges(),flush=True)
    return pd.DataFrame(out),pd.DataFrame(actions)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo-mtx',type=Path);p.add_argument('--max-n',type=int,default=800);p.add_argument('--max-edges',type=int,default=12000);p.add_argument('--train-graphs',type=int,default=30);p.add_argument('--steps',type=int,default=40);p.add_argument('--reps',type=int,default=2);p.add_argument('--seeds',type=int,default=2);p.add_argument('--max-test',type=int,default=200);args=p.parse_args()
    torch.set_num_threads(1)
    if args.repo_mtx:
        gs,audit=discover(args.repo_mtx,args.max_n,args.max_edges)
        pd.DataFrame(audit).to_csv(ROOT/'inventory.csv',index=False)
        if not gs:raise SystemExit('No eligible real matrices')
        gs=gs[:args.max_test]
    else:
        gs=bundled()
        pd.DataFrame([{'path':x[0],'family':x[1],'n':len(x[2]),'edges':x[2].number_of_edges(),'status':'included','reason':'bundled previously retrieved'} for x in gs]).to_csv(ROOT/'inventory.csv',index=False)
    assert len(set(signature(g) for _,_,g in gs))==len(gs),'duplicate topologies in evaluation'
    training={};agents={}
    for i in range(args.seeds):
        seed=47+31*i; agents[seed]={}
        for mode in ['original','rich','drop_history','drop_critical']:
            ag,duration,_=train_ablation(seed,mode,args.train_graphs,args.steps)
            agents[seed][mode]=ag
            training[(seed,mode)]={'seed':seed,'mode':mode,'train_seconds':duration,'updates':ag.updates,'train_graphs':args.train_graphs}
            print('TRAINED',seed,mode,round(duration,2),flush=True)
    results,actions=evaluate(agents,gs,args.seeds,args.reps,args.steps)
    results.to_csv(ROOT/'results.csv',index=False);actions.to_csv(ROOT/'actions.csv',index=False)
    pd.DataFrame(list(training.values())).to_csv(ROOT/'training.csv',index=False)
    summary=results.groupby('method').agg(mean_bw=('bw','mean'),mean_gain=('bw',lambda x:0),mean_refinement_ms=('refinement_sec',lambda s:1000*s.mean()),mean_inference_ms=('total_inference_sec',lambda s:1000*s.mean()),matrices=('graph','nunique')).reset_index()
    summary['mean_gain']=results.groupby('method').apply(lambda a:(a.rcm-a.bw).mean(),include_groups=False).to_numpy()
    summary.to_csv(ROOT/'summary.csv',index=False)
    per=results.groupby(['graph','family','method']).agg(mean_bw=('bw','mean'),rcm=('rcm','mean')).reset_index()
    per.to_csv(ROOT/'matrix_summary.csv',index=False)
    table=per.pivot(index='graph',columns='method',values='mean_bw')
    comp={}
    for method in ['DQN_original','DQN_rich','DQN_rich_adaptive','Discounted_UCB']:
        comp[method]={}
        for ref in ['RCM','Random','Fixed_adjacent','DQN_original']:
            if method==ref:continue
            diff=table[method]-table[ref]
            comp[method][ref]={'mean_delta_bw':float(diff.mean()),'wins':int((diff<0).sum()),'ties':int((diff==0).sum()),'losses':int((diff>0).sum()),'wilcoxon_p_exploratory':float(wilcoxon(diff).pvalue) if np.any(diff!=0) else None}
    (ROOT/'diagnostics.json').write_text(json.dumps({'source':'github_local_copy' if args.repo_mtx else '5 previously downloaded github graphs only','matrices':len(gs),'family_groups':len(set(x[1] for x in gs)),'training_only':'synthetic','steps':args.steps,'seeds':args.seeds,'reps':args.reps,'notes':['Equal number of refinement proposals, not equal CPU time','RCM included in reported total_inference_sec; DQN training separately','Family label is inferred from filenames, not curated SuiteSparse classification','Not a leave-family-out test: real graphs are never used during training','For 50-100 repository matrices supply cloned repo via --repo-mtx','Single/multiple non-normality p-values are exploratory'],'comparisons':comp},indent=2))
    print('SUMMARY\n'+summary.to_string(index=False),flush=True)
    print('COMPARISONS\n'+json.dumps(comp,indent=2),flush=True)

if __name__=='__main__': main()
