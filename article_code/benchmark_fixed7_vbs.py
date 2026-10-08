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

from benchmark_ablation import CENTRALITIES
from centralities import get_centrality_node, centrality_heuristic
from modules.graph.Grafo import GrafoListaAdj
from modules.utils import read_Instances
from modules.utils.handle_labels import Bf_graph


def load_graph(path):
    nnodes, nedges, edges, _, _, _ = read_Instances.load_instance(path)
    G = nx.Graph()
    G.add_nodes_from(range(1, nnodes + 1))
    G.add_edges_from(edges)
    g = GrafoListaAdj()
    g.DefinirN(nnodes, VizinhancaDuplamenteLigada=True)
    for u, v in edges:
        g.AdicionarAresta(u, v)
    positions = {node: i for i, node in enumerate(G.nodes())}
    original = max((abs(positions[u] - positions[v]) for u, v in G.edges()), default=0)
    return nnodes, nedges, G, g, original


def raw_rcm(G, g):
    t0 = time.perf_counter()
    order = list(nx.utils.reverse_cuthill_mckee_ordering(G))
    labels = {node: i + 1 for i, node in enumerate(order)}
    bw = int(Bf_graph(g, labels))
    return bw, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--decisions", type=int, default=360)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    path = os.path.join(a.data_dir, a.instance)
    nnodes, nedges, G, g, original = load_graph(path)

    maps = {}
    cent_times = {}
    for name, cfg in CENTRALITIES.items():
        t0 = time.perf_counter()
        maps[name] = get_centrality_node(G, cfg)
        cent_times[name] = time.perf_counter() - t0

    rcm_bw, rcm_time = raw_rcm(G, g)
    rows = []
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        row = {
            "instance": a.instance,
            "seed": seed,
            "nodes": nnodes,
            "edges": nedges,
            "original_bw": original,
            "raw_rcm_bw": rcm_bw,
            "rcm_time_s": rcm_time,
        }
        vals = []
        for name in CENTRALITIES:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            t0 = time.perf_counter()
            cost, _, _, _ = centrality_heuristic(
                graph=g,
                centrality_values=maps[name],
                cent_str=name,
                alpha=0.3,
                iter_max=a.decisions * 15,
                centralities=CENTRALITIES,
            )
            search_t = time.perf_counter() - t0
            best = int(min(original, cost))
            key = name.lower().replace(" ", "_")
            row[key + "_best"] = best
            row[key + "_centrality_time_s"] = cent_times[name]
            row[key + "_search_time_s"] = search_t
            row[key + "_total_time_s"] = cent_times[name] + search_t
            vals.append(best)
        row["vbs_best"] = min(vals)
        rows.append(row)
        print(a.instance, seed, "B0", original, "RCM(raw)", rcm_bw, "VBS", row["vbs_best"])

    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
