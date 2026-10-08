import argparse
import csv
import os
import random
import sys
import time

import networkx as nx
import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark_ablation import CENTRALITIES, build
from centralities import centrality_heuristic
from dqn_v2 import DQNSearchV2
from modules.utils.handle_labels import Bf_graph


def raw_rcm(G, g):
    order = list(nx.utils.reverse_cuthill_mckee_ordering(G))
    labels = {node: i + 1 for i, node in enumerate(order)}
    return int(Bf_graph(g, labels))


def pseudo_peripheral_node(G):
    if G.number_of_nodes() == 0:
        return None
    if nx.is_connected(G):
        H = G
    else:
        largest = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest)

    current = min(H.nodes(), key=lambda v: (H.degree(v), v))
    last_ecc = -1
    for _ in range(max(2, H.number_of_nodes())):
        lengths = nx.single_source_shortest_path_length(H, current)
        ecc = max(lengths.values(), default=0)
        farthest = [v for v, d in lengths.items() if d == ecc]
        candidate = min(farthest, key=lambda v: (H.degree(v), v))
        if ecc <= last_ecc:
            return current
        last_ecc = ecc
        current = candidate
    return current


def eval_fixed(g, maps, name, decisions, original, start_node=None):
    constructions = decisions * 15
    cost, _, _, _ = centrality_heuristic(
        graph=g,
        centrality_values=maps[name],
        cent_str=name,
        alpha=0.3,
        iter_max=constructions,
        centralities=CENTRALITIES,
        start_node=start_node,
    )
    return int(min(original, cost))


def run_dqn(g, maps, original, train_steps, eval_steps, seed, start_node=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    solver = DQNSearchV2(
        graph=g,
        centrality_maps=maps,
        centralities=CENTRALITIES,
        initial_bw=original,
        train_steps=train_steps,
        eval_steps=eval_steps,
        samples_per_action=15,
        start_node=start_node,
    )
    t0 = time.perf_counter()
    train_state, _ = solver.train()
    eval_state, eval_hist = solver.evaluate_greedy()
    elapsed = time.perf_counter() - t0
    total_best = int(min(train_state.best_bw, eval_state.best_bw))
    freq = {
        name: sum(x["centrality"] == name for x in eval_hist) / max(1, len(eval_hist))
        for name in solver.names
    }
    return total_best, elapsed, freq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--eval-steps", type=int, default=60)
    p.add_argument("--output", required=True)
    p.add_argument("--include-fixed", action="store_true")
    a = p.parse_args()

    path = os.path.join(a.data_dir, a.instance)
    nnodes, nedges, G, g, original, maps = build(path)
    peripheral = pseudo_peripheral_node(G)
    rcm = raw_rcm(G, g)
    total_decisions = a.train_steps + a.eval_steps
    rows = []

    for seed in range(a.seed_start, a.seed_start + a.seeds):
        std, std_t, std_freq = run_dqn(
            g, maps, original, a.train_steps, a.eval_steps, seed, None
        )
        per, per_t, per_freq = run_dqn(
            g, maps, original, a.train_steps, a.eval_steps, seed, peripheral
        )

        row = {
            "instance": a.instance,
            "seed": seed,
            "nodes": nnodes,
            "edges": nedges,
            "original_bw": original,
            "raw_rcm_bw": rcm,
            "pseudo_peripheral_node": peripheral,
            "dqn_standard": std,
            "dqn_peripheral": per,
            "dqn_standard_time_s": std_t,
            "dqn_peripheral_time_s": per_t,
        }

        for name in CENTRALITIES:
            key = name.lower().replace(" ", "_")
            row["evalfreq_standard_" + key] = std_freq[name]
            row["evalfreq_peripheral_" + key] = per_freq[name]

        if a.include_fixed:
            for name in CENTRALITIES:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                row["fixed_" + name.lower().replace(" ", "_")] = eval_fixed(
                    g, maps, name, total_decisions, original, None
                )
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                row["peripheral_" + name.lower().replace(" ", "_")] = eval_fixed(
                    g, maps, name, total_decisions, original, peripheral
                )
            fixed_cols = [row["fixed_" + n.lower().replace(" ", "_")] for n in CENTRALITIES]
            per_cols = [row["peripheral_" + n.lower().replace(" ", "_")] for n in CENTRALITIES]
            row["vbs_fixed"] = min(fixed_cols)
            row["vbs_peripheral"] = min(per_cols)

        rows.append(row)
        print(a.instance, seed, "B0", original, "RCM", rcm,
              "DQN", std, "DQN-per", per)

    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
