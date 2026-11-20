"""Portable GitHub Actions entrypoint for the BandWidths RCM benchmark.

The tested algorithms are the same four-operation environments, DQN variants,
and Discounted-UCB as used in the exploratory experiments, with real graphs
kept out of synthetic DQN training.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in ('repo_benchmark_audit', 'rcm_ablation', 'adaptive_rcm', 'dynamic_rcm'):
    sys.path.insert(0, str(HERE / p))

import numpy as np
import pandas as pd
import torch
from benchmark import discover, graph_from_mtx, signature, evaluate
from run import train as train_ablation
from experiment import DQN

MODES = ('original', 'rich', 'drop_history', 'drop_critical')


def save_json(obj, path):
    p = Path(path);p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def inventory(args):
    base = Path(args.repo).resolve()
    graphs, audit = discover(base, args.max_n, args.max_edges)
    included = [r for r in audit if r['status'] == 'included']
    assert len(graphs) == len(included)
    chosen = []
    for (key, family, g), record in zip(graphs, included):
        chosen.append(dict(key=key, family=family, path=str(Path(record['path']).relative_to(base)),
                           n=len(g), edges=g.number_of_edges(), signature=signature(g)))
    # Pilot uses small eligible examples; all mode selects the entire eligible dataset.
    if args.mode == 'pilot':
        chosen = sorted(chosen, key=lambda r:(r['n'],r['edges']))[:args.pilot_count]
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(out / 'inventory.csv', index=False)
    save_json(dict(mode=args.mode, repo=str(base), count=len(chosen),
                   max_n=args.max_n, max_edges=args.max_edges, matrices=chosen), out / 'manifest.json')
    if not chosen:raise SystemExit('No eligible matrices')
    print('ELIGIBLE', len(graphs), 'SELECTED', len(chosen), 'FILES', len(audit),flush=True)


def train(args):
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    seed = 47 + 31*args.seed_index
    records=[]
    for mode in MODES:
        agent, seconds, _ = train_ablation(seed, mode, args.train_graphs, args.steps)
        torch.save(agent.net.state_dict(), out / f'{mode}.pth')
        records.append(dict(seed_index=args.seed_index, seed=seed, steps=args.steps,
                            mode=mode, train_graphs=args.train_graphs,
                            training_seconds=seconds, updates=agent.updates))
        print('TRAINED', mode, seconds,flush=True)
    pd.DataFrame(records).to_csv(out / 'training.csv', index=False)


def eval_batch(args):
    manifest=json.loads(Path(args.manifest).read_text())
    all_matrices=manifest['matrices']
    batch=[row for i,row in enumerate(all_matrices) if i%args.shards==args.shard]
    if not batch:
        print('EMPTY SHARD',args.shard,flush=True)
        return
    repo=Path(args.repo).resolve()
    graphs=[]
    for item in batch:
        g=graph_from_mtx(repo/item['path'])
        if signature(g)!=item['signature']:
            raise RuntimeError('Changed matrix or manifest mismatch: '+item['path'])
        graphs.append((item['key'],item['family'],g))
    seed=47+31*args.seed_index
    agents={seed:{}}
    for mode in MODES:
        ag=DQN(seed,10 if mode=='original' else 29)
        # models produced by train job in same workflow; no pickle objects loaded
        ag.net.load_state_dict(torch.load(Path(args.models)/f'{mode}.pth',map_location='cpu',weights_only=True))
        ag.net.eval()
        agents[seed][mode]=ag
    rows, actions=evaluate(agents,graphs,1,args.reps,args.steps)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    suffix=f's{args.seed_index}_b{args.steps}_part{args.shard}'
    rows.to_csv(out/f'results_{suffix}.csv',index=False)
    actions.to_csv(out/f'actions_{suffix}.csv',index=False)
    print('COMPLETE',suffix, 'graphs',len(graphs),'rows',len(rows),flush=True)


def aggregate(args):
    import scipy.stats as stats
    import glob
    src=Path(args.input)
    resfiles=list(src.rglob('results_*.csv'))
    if not resfiles:raise SystemExit('No evaluation CSV files were downloaded')
    df=pd.concat((pd.read_csv(f) for f in resfiles),ignore_index=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'all_results.csv',index=False)
    grp=df.groupby('method',as_index=False).agg(mean_bandwidth=('bw','mean'),
       median_bandwidth=('bw','median'),mean_rcm=('rcm','mean'),
       mean_total_seconds=('total_inference_sec','mean'),
       graphs=('graph','nunique'),evaluations=('bw','size'))
    grp['mean_improvement']=grp.mean_rcm-grp.mean_bandwidth
    grp.to_csv(out/'summary.csv',index=False)
    fam=df.groupby(['family','method'],as_index=False).agg(mean_bandwidth=('bw','mean'),
       mean_total_seconds=('total_inference_sec','mean'),graphs=('graph','nunique'))
    fam.to_csv(out/'family_summary.csv',index=False)
    per=df.groupby(['graph','family','method'],as_index=False).agg(mean_bw=('bw','mean'))
    per.to_csv(out/'per_graph.csv',index=False)
    pivot=per.pivot(index='graph',columns='method',values='mean_bw')
    cmp=[]
    for a in ('DQN_original','DQN_rich','DQN_no_history','DQN_no_critical','DQN_rich_adaptive','Discounted_UCB'):
        for b in ('RCM','Random','Fixed_adjacent'):
            if a not in pivot or b not in pivot:continue
            d=(pivot[a]-pivot[b]).dropna()
            p= float(stats.wilcoxon(d).pvalue) if len(d)>0 and (d!=0).any() else None
            cmp.append(dict(method=a,reference=b,graphs=len(d),mean_delta=float(d.mean()),
                            wins=int((d<0).sum()),losses=int((d>0).sum()),ties=int((d==0).sum()),wilcoxon_p_exploratory=p))
    pd.DataFrame(cmp).to_csv(out/'comparisons.csv',index=False)
    # A family is the unit of analysis for this additional summary; family labels are heuristic.
    by_family=fam.pivot(index='family',columns='method',values='mean_bandwidth')
    fcmp=[]
    for a in ('DQN_original','DQN_rich','Discounted_UCB'):
        if a in by_family and 'Fixed_adjacent' in by_family:
            dif=(by_family[a]-by_family['Fixed_adjacent']).dropna()
            fcmp.append(dict(method=a,reference='Fixed_adjacent',families=len(dif),
                             mean_delta=float(dif.mean()),wins=int((dif<0).sum()),
                             losses=int((dif>0).sum()),ties=int((dif==0).sum())))
    pd.DataFrame(fcmp).to_csv(out/'family_comparisons.csv',index=False)
    print(grp.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest='command',required=True)
    inv=s.add_parser('inventory');inv.add_argument('--repo',required=True);inv.add_argument('--output',required=True)
    inv.add_argument('--mode',choices=('pilot','full'),default='pilot');inv.add_argument('--pilot-count',type=int,default=5)
    inv.add_argument('--max-n',type=int,default=800);inv.add_argument('--max-edges',type=int,default=12000)
    tr=s.add_parser('train');tr.add_argument('--seed-index',type=int,required=True);tr.add_argument('--steps',type=int,required=True)
    tr.add_argument('--train-graphs',type=int,default=55);tr.add_argument('--output',required=True)
    ev=s.add_parser('evaluate');ev.add_argument('--repo',required=True);ev.add_argument('--manifest',required=True)
    ev.add_argument('--models',required=True);ev.add_argument('--seed-index',type=int,required=True)
    ev.add_argument('--steps',type=int,required=True);ev.add_argument('--reps',type=int,default=2)
    ev.add_argument('--shard',type=int,required=True);ev.add_argument('--shards',type=int,required=True)
    ev.add_argument('--output',required=True)
    ag=s.add_parser('aggregate');ag.add_argument('--input',required=True);ag.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(1)
    {'inventory':inventory,'train':train,'evaluate':eval_batch,'aggregate':aggregate}[a.command](a)

if __name__=='__main__':main()
