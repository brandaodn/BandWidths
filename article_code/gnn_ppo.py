import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.utils import read_Instances


def canonical_bandwidth(G, order):
    pos = {node: i for i, node in enumerate(order)}
    return max((abs(pos[u] - pos[v]) for u, v in G.edges()), default=0)


def rcm_order(G):
    return list(nx.utils.reverse_cuthill_mckee_ordering(G))


def normalized_adjacency(G, nodes, device):
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = torch.zeros((n, n), dtype=torch.float32, device=device)
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(n, device=device)
    deg = A.sum(dim=1)
    d = deg.clamp(min=1e-8).pow(-0.5)
    return d[:, None] * A * d[None, :]


def node_features(G, nodes, order, initial_order, device):
    n = len(nodes)
    pos = {v: i for i, v in enumerate(order)}
    init = {v: i for i, v in enumerate(initial_order)}
    maxdeg = max((G.degree(v) for v in nodes), default=1)
    bw = max(1, canonical_bandwidth(G, order))
    feats = []
    for v in nodes:
        local = max((abs(pos[v] - pos[u]) for u in G.neighbors(v)), default=0)
        feats.append([
            G.degree(v) / max(1, maxdeg),
            pos[v] / max(1, n - 1),
            init[v] / max(1, n - 1),
            local / bw,
        ])
    return torch.tensor(feats, dtype=torch.float32, device=device)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden=64, out_dim=64):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)

    def forward(self, x, A):
        h = torch.relu(self.lin1(A @ x))
        return torch.relu(self.lin2(A @ h))


class ActorCritic(nn.Module):
    """Score a variable set of swap/relocation actions."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.encoder = GCNEncoder(4, 64, embed_dim)
        self.actor = nn.Sequential(
            nn.Linear(embed_dim * 2 + 3, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, A, order_idx, actions):
        h = self.encoder(x, A)
        ordered = h[order_idx]

        src = actions[:, 1].long()
        dst = actions[:, 2].long()
        action_meta = actions[:, [0, 3, 4]]
        pair = torch.cat([ordered[src], ordered[dst], action_meta], dim=1)
        logits = self.actor(pair).squeeze(-1)
        value = self.critic(h.mean(dim=0)).squeeze(-1)
        return logits, value


@dataclass
class Transition:
    x: torch.Tensor
    order_idx: torch.Tensor
    actions: torch.Tensor
    action: int
    logp: float
    reward: float
    done: bool
    value: float


class BandwidthEnv:
    """
    Local-improvement environment.

    Candidate actions are restricted to high-impact vertices and geometric
    position offsets. This gives non-adjacent swaps and relocations without
    enumerating all O(n^2) pairs.
    """

    def __init__(
        self,
        G,
        device,
        critical_nodes=16,
        offsets=(1, 2, 4, 8, 16, 32),
        enable_swap=True,
        enable_relocation=True,
    ):
        self.G = G
        self.nodes = list(G.nodes())
        self.device = device
        self.A = normalized_adjacency(G, self.nodes, device)
        self.initial_order = rcm_order(G)
        self.critical_nodes = critical_nodes
        self.offsets = tuple(offsets)
        self.enable_swap = enable_swap
        self.enable_relocation = enable_relocation
        self.action_specs = []
        self.reset()

    def reset(self):
        self.order = list(self.initial_order)
        self.initial_bw = canonical_bandwidth(self.G, self.order)
        self.current_bw = self.initial_bw
        self.best_bw = self.current_bw
        self.best_order = list(self.order)
        return self.state()

    def _critical_edges(self):
        """Return edges that currently attain, or nearly attain, the bandwidth."""
        pos = {v: i for i, v in enumerate(self.order)}
        spans = []
        for u, v in self.G.edges():
            span = abs(pos[u] - pos[v])
            spans.append((span, u, v))

        if not spans:
            return []

        spans.sort(reverse=True)
        max_span = spans[0][0]
        threshold = max(1, max_span - 1)

        critical = [(u, v, span) for span, u, v in spans if span >= threshold]
        return critical[: self.critical_nodes]

    def _build_action_candidates(self):
        n = len(self.order)
        pos = {v: i for i, v in enumerate(self.order)}
        specs = set()

        # Focus on endpoints of edges that currently determine (or almost
        # determine) the bandwidth. Each endpoint can move toward the other
        # endpoint or perform larger exploratory jumps.
        for u, v, _ in self._critical_edges():
            iu, iv = pos[u], pos[v]
            endpoint_pairs = [(iu, iv), (iv, iu)]

            for i, target in endpoint_pairs:
                # Direct critical-edge repair candidate.
                if i != target:
                    if self.enable_swap:
                        specs.add((0, i, target))
                    if self.enable_relocation:
                        specs.add((1, i, target))

                direction = 1 if target > i else -1
                distance_to_target = abs(target - i)

                for offset in self.offsets:
                    # Prefer moving toward the opposite critical endpoint.
                    step = min(offset, distance_to_target)
                    j = i + direction * step
                    if 0 <= j < n and j != i:
                        if self.enable_swap:
                            specs.add((0, i, j))
                        if self.enable_relocation:
                            specs.add((1, i, j))

                    # Also keep a small symmetric exploration component.
                    for sign in (-1, 1):
                        j = i + sign * offset
                        if j < 0 or j >= n or j == i:
                            continue
                        if self.enable_swap:
                            specs.add((0, i, j))
                        if self.enable_relocation:
                            specs.add((1, i, j))

        # Fallback for edgeless/small graphs.
        if not specs and n >= 2:
            specs.add((0, 0, 1))

        self.action_specs = sorted(specs)
        denom = max(1, n - 1)
        rows = []
        for kind, i, j in self.action_specs:
            distance = abs(j - i) / denom
            direction = 1.0 if j > i else -1.0
            rows.append([float(kind), float(i), float(j), distance, direction])

        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def state(self):
        x = node_features(
            self.G, self.nodes, self.order, self.initial_order, self.device
        )
        node_to_idx = {v: i for i, v in enumerate(self.nodes)}
        order_idx = torch.tensor(
            [node_to_idx[v] for v in self.order],
            dtype=torch.long,
            device=self.device,
        )
        actions = self._build_action_candidates()
        return x, order_idx, actions

    def step(self, action_index):
        if len(self.order) < 2:
            return self.state(), 0.0, True

        action_index = int(max(0, min(action_index, len(self.action_specs) - 1)))
        kind, i, j = self.action_specs[action_index]
        old = self.current_bw

        if kind == 0:
            self.order[i], self.order[j] = self.order[j], self.order[i]
        else:
            node = self.order.pop(i)
            self.order.insert(j, node)

        self.current_bw = canonical_bandwidth(self.G, self.order)
        reward = (old - self.current_bw) / max(1, self.initial_bw)

        if self.current_bw < self.best_bw:
            self.best_bw = self.current_bw
            self.best_order = list(self.order)

        return self.state(), float(reward), False


class PPOTrainer:
    def __init__(
        self,
        G,
        lr=3e-4,
        gamma=.99,
        gae_lambda=.95,
        clip=.2,
        epochs=4,
        device=None,
        critical_nodes=16,
        offsets=(1, 2, 4, 8, 16, 32),
        enable_swap=True,
        enable_relocation=True,
    ):
        self.device = device or torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.env = BandwidthEnv(
            G,
            self.device,
            critical_nodes=critical_nodes,
            offsets=offsets,
            enable_swap=enable_swap,
            enable_relocation=enable_relocation,
        )
        self.model = ActorCritic().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs

    def collect(self, steps):
        transitions = []
        x, order_idx, actions = self.env.reset()

        for t in range(steps):
            with torch.no_grad():
                logits, value = self.model(
                    x, self.env.A, order_idx, actions
                )
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            (next_x, next_order_idx, next_actions), reward, _ = self.env.step(
                action.item()
            )
            done = t == steps - 1
            transitions.append(
                Transition(
                    x=x.detach(),
                    order_idx=order_idx.detach(),
                    actions=actions.detach(),
                    action=int(action.item()),
                    logp=float(logp.item()),
                    reward=float(reward),
                    done=done,
                    value=float(value.item()),
                )
            )
            x, order_idx, actions = next_x, next_order_idx, next_actions

        return transitions, self.env.best_bw, self.env.best_order

    def _advantages(self, transitions):
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions] + [0.0]
        advantages = [0.0] * len(transitions)
        gae = 0.0

        for i in reversed(range(len(transitions))):
            delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[i] = gae

        returns = [
            advantages[i] + values[i] for i in range(len(transitions))
        ]
        a = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        if len(a) > 1 and float(a.std()) > 1e-8:
            a = (a - a.mean()) / (a.std() + 1e-8)

        return a, torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )

    def update(self, transitions):
        advantages, returns = self._advantages(transitions)
        old_logp = torch.tensor(
            [t.logp for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        for _ in range(self.epochs):
            losses = []
            for i, t in enumerate(transitions):
                logits, value = self.model(
                    t.x, self.env.A, t.order_idx, t.actions
                )
                dist = Categorical(logits=logits)
                action = torch.tensor(t.action, device=self.device)
                logp = dist.log_prob(action)
                ratio = torch.exp(logp - old_logp[i])

                s1 = ratio * advantages[i]
                s2 = (
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                    * advantages[i]
                )
                actor_loss = -torch.min(s1, s2)
                critic_loss = 0.5 * (value - returns[i]).pow(2)
                entropy = dist.entropy()
                losses.append(
                    actor_loss + critic_loss - 0.01 * entropy
                )

            loss = torch.stack(losses).mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )
            self.opt.step()

    def train(self, episodes=20, steps=30, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        global_best = self.env.initial_bw
        global_order = list(self.env.initial_order)
        history = []

        for ep in range(episodes):
            transitions, bw, order = self.collect(steps)
            self.update(transitions)

            if bw < global_best:
                global_best = bw
                global_order = list(order)

            history.append(
                {
                    "episode": ep + 1,
                    "best_bw": int(bw),
                    "global_best_bw": int(global_best),
                }
            )

        return {
            "rcm_bw": int(self.env.initial_bw),
            "best_bw": int(global_best),
            "best_order": global_order,
            "history": history,
        }


def load_graph(path):
    nnodes, _, edges, _, _, _ = read_Instances.load_instance(path)
    G = nx.Graph()
    G.add_nodes_from(range(1, nnodes + 1))
    G.add_edges_from(edges)
    return G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instance")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--critical-nodes", type=int, default=16)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    G = load_graph(a.instance)
    trainer = PPOTrainer(G, critical_nodes=a.critical_nodes)
    result = trainer.train(
        episodes=a.episodes,
        steps=a.steps,
        seed=a.seed,
    )
    result.update(
        {
            "instance": os.path.basename(a.instance),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }
    )
    print(json.dumps(result, indent=2))

    if a.output:
        with open(a.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
