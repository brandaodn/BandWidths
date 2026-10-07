import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gnn_ppo import PPOTrainer, load_graph


def run_policy(G, seed, episodes, steps, mode):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode == "adjacent":
        trainer = PPOTrainer(
            G,
            device=torch.device("cpu"),
            critical_nodes=max(1, G.number_of_nodes()),
            offsets=(1,),
            enable_swap=True,
            enable_relocation=False,
        )
    elif mode == "mixed":
        trainer = PPOTrainer(
            G,
            device=torch.device("cpu"),
            critical_nodes=16,
            offsets=(1, 2, 4, 8, 16, 32),
            enable_swap=True,
            enable_relocation=True,
        )
    else:
        raise ValueError(mode)

    start = time.perf_counter()
    result = trainer.train(episodes=episodes, steps=steps, seed=seed)
    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--output", required=True)
    p.add_argument("instances", nargs="+")
    a = p.parse_args()

    rows = []
    for name in a.instances:
        G = load_graph(os.path.join(a.data_dir, name))
        for seed in range(1, a.seeds + 1):
            for mode in ("adjacent", "mixed"):
                result, elapsed = run_policy(
                    G, seed, a.episodes, a.steps, mode
                )
                rows.append(
                    {
                        "instance": name,
                        "seed": seed,
                        "mode": mode,
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "rcm_bw": result["rcm_bw"],
                        "best_bw": result["best_bw"],
                        "improvement": result["rcm_bw"] - result["best_bw"],
                        "improvement_pct": (
                            0.0
                            if result["rcm_bw"] == 0
                            else 100.0
                            * (result["rcm_bw"] - result["best_bw"])
                            / result["rcm_bw"]
                        ),
                        "elapsed_s": elapsed,
                    }
                )

    fields = list(rows[0].keys())
    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
