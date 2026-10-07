import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import networkx as nx

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import Agent
from enviroment import Env
from centralities import get_centrality_node
from modules.graph.Grafo import GrafoListaAdj
from modules.utils import read_Instances


CENTRALITIES = {
    "Degree": {"func": nx.degree_centrality, "args": {}, "reverse": True},
    "Closeness": {"func": nx.closeness_centrality, "args": {}, "reverse": True},
    "Betweenness": {"func": nx.betweenness_centrality, "args": {}, "reverse": True},
    "Eigenvector": {"func": nx.eigenvector_centrality, "args": {"max_iter": 1000, "tol": 1e-6}, "reverse": True},
    "Katz Centrality": {"func": nx.katz_centrality, "args": {"alpha": 0.005, "beta": 1.0, "max_iter": 1000, "tol": 1e-6}, "reverse": True},
    "PageRank": {"func": nx.pagerank, "args": {"alpha": 0.85}, "reverse": True},
    "Harmonic Centrality": {"func": nx.harmonic_centrality, "args": {}, "reverse": True},
}


def run_instance(path, iterations, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nnodes, nedges, edges, _, _, _ = read_Instances.load_instance(path)

    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(1, nnodes + 1))
    graph_nx.add_edges_from(edges)

    graph = GrafoListaAdj()
    graph.DefinirN(nnodes, VizinhancaDuplamenteLigada=True)
    for u, v in edges:
        graph.AdicionarAresta(u, v)

    positions = {node: idx for idx, node in enumerate(graph_nx.nodes())}
    original_bw = max((abs(positions[u] - positions[v]) for u, v in graph_nx.edges()), default=0)
    centrality_maps = {
        name: get_centrality_node(graph_nx, config)
        for name, config in CENTRALITIES.items()
    }
    names = list(CENTRALITIES.keys())

    agent = Agent(
        learning_rate=0.001,
        gamma=0.9,
        epsilon=1.0,
        eps_min=0.01,
        eps_dec=0.995,
        n_movements=len(names),
        n_actions=len(names),
        n_states=4,
        deep=True,
    )

    # Use the same initial reference for both versions so the comparison
    # targets implementation behavior rather than the old hard-coded 100000.
    env = Env(original_bw)
    state = env.get_initial_state()
    best_bw = original_bw

    start = time.perf_counter()
    for i in range(iterations):
        action = agent.choose_action(state)
        centrality = names[action]
        info = env.step(
            graph=graph,
            centrality_values=centrality_maps[centrality],
            cent_str=centrality,
            centralities=CENTRALITIES,
        )
        new_state = [
            info["n_steps"],
            info["gap"],
            info["reward"],
            info["bandwidth"],
        ]
        try:
            agent.learn(state, action, info["reward"], new_state, done=(i == iterations - 1))
        except TypeError:
            # Compatibility with the old Agent.learn signature.
            agent.learn(state, action, info["reward"], new_state)
        state = new_state
        best_bw = min(best_bw, info["bandwidth"])

    elapsed = time.perf_counter() - start
    return {
        "instance": os.path.basename(path),
        "nodes": nnodes,
        "edges": nedges,
        "original_bw": original_bw,
        "best_bw": best_bw,
        "reduction_pct": 0.0 if original_bw == 0 else 100.0 * (original_bw - best_bw) / original_bw,
        "elapsed_s": float(elapsed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", required=True)
    parser.add_argument("instances", nargs="+")
    args = parser.parse_args()

    rows = []
    for name in args.instances:
        rows.append(
            run_instance(
                os.path.join(args.data_dir, name),
                iterations=args.iterations,
                seed=args.seed,
            )
        )

    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value))

    print(json.dumps(rows, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)))


if __name__ == "__main__":
    main()
