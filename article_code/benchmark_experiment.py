import argparse
import csv
import json
import os
from benchmark_single import run_instance

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True)
    p.add_argument("--iterations",type=int,default=30)
    p.add_argument("--seeds",type=int,default=30)
    p.add_argument("--output",required=True)
    p.add_argument("instances",nargs="+")
    a=p.parse_args()

    rows=[]
    for seed in range(1,a.seeds+1):
        for name in a.instances:
            row=run_instance(
                os.path.join(a.data_dir,name),
                iterations=a.iterations,
                seed=seed,
            )
            row["seed"]=seed
            rows.append(row)

    fields=["instance","seed","nodes","edges","original_bw","best_bw","reduction_pct","elapsed_s"]
    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({"runs":len(rows),"output":a.output},indent=2))

if __name__=="__main__":
    main()
