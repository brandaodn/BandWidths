import argparse
import json
import math
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


def build_edge_index(G, nodes, device):
    idx = {v: i for i, v in enumerate(nodes)}
    src, dst = [], []
    for u, v in G.edges():
        iu, iv = idx[u], idx[v]
        src.extend([iu, iv])
        dst.extend([iv, iu])
    for i in range(len(nodes)):
        src.append(i)
        dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long, device=device)


class SparseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        n = x.shape[0]
        deg = torch.bincount(dst, minlength=n).float().clamp(min=1.0)
        z = self.lin(x)
        norm = (deg[src] * deg[dst]).rsqrt().unsqueeze(1)
        msg = z[src] * norm
        out = torch.zeros((n, z.shape[1]), dtype=z.dtype, device=z.device)
        out.index_add_(0, dst, msg)
        return out


class SparseGCNEncoder(nn.Module):
    def __init__(self, in_dim=8, hidden=64, out_dim=64):
        super().__init__()
        self.g1 = SparseGCNLayer(in_dim, hidden)
        self.g2 = SparseGCNLayer(hidden, out_dim)

    def forward(self, x, edge_index):
        h = torch.relu(self.g1(x, edge_index))
        return torch.relu(self.g2(h, edge_index))


class ActorCriticV2(nn.Module):
    def __init__(self, node_dim=8, embed_dim=64, global_dim=7):
        super().__init__()
        self.encoder = SparseGCNEncoder(node_dim, 64, embed_dim)
        actor_in = embed_dim * 3 + global_dim + 3
        self.actor = nn.Sequential(
            nn.Linear(actor_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim + global_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x, edge_index, order_idx, actions, global_state):
        h = self.encoder(x, edge_index)
        ordered = h[order_idx]
        graph_emb = h.mean(dim=0)

        src = actions[:, 1].long()
        dst = actions[:, 2].long()
        action_meta = actions[:, [0, 3, 4]]
        g = graph_emb.unsqueeze(0).expand(actions.shape[0], -1)
        gs = global_state.unsqueeze(0).expand(actions.shape[0], -1)
        pair = torch.cat(
            [ordered[src], ordered[dst], g, gs, action_meta], dim=1
        )
        logits = self.actor(pair).squeeze(-1)
        value = self.critic(
            torch.cat([graph_emb, global_state], dim=0)
        ).squeeze(-1)
        return logits, value


@dataclass
class Transition:
    x: torch.Tensor
    order_idx: torch.Tensor
    actions: torch.Tensor
    global_state: torch.Tensor
    action: int
    logp: float
    reward: float
    done: bool
    value: float


class BandwidthEnvV2:
    def __init__(
        self,
        G,
        device,
        horizon=40,
        critical_nodes=24,
        offsets=(1, 2, 4, 8, 16, 32, 64),
        enable_swap=True,
        enable_relocation=True,
    ):
        self.G = G
        self.nodes = list(G.nodes())
        self.device = device
        self.horizon = max(1, int(horizon))
        self.edge_index = build_edge_index(G, self.nodes, device)
        self.initial_order = rcm_order(G)
        self.critical_nodes = critical_nodes
        self.offsets = tuple(offsets)
        self.enable_swap = enable_swap
        self.enable_relocation = enable_relocation

        maxdeg = max((G.degree(v) for v in self.nodes), default=1)
        core = nx.core_number(G) if G.number_of_edges() else {v: 0 for v in self.nodes}
        maxcore = max(core.values(), default=1)
        clustering = nx.clustering(G)
        self.static = {
            v: (
                G.degree(v) / max(1, maxdeg),
                core.get(v, 0) / max(1, maxcore),
                float(clustering.get(v, 0.0)),
            )
            for v in self.nodes
        }
        self.action_specs = []
        self.reset()

    def reset(self):
        self.order = list(self.initial_order)
        self.initial_bw = canonical_bandwidth(self.G, self.order)
        self.current_bw = self.initial_bw
        self.best_bw = self.initial_bw
        self.best_order = list(self.order)
        self.step_count = 0
        self.steps_since_improvement = 0
        return self.state()

    def _positions(self):
        return {v: i for i, v in enumerate(self.order)}

    def _critical_edges(self):
        pos = self._positions()
        spans = [(abs(pos[u] - pos[v]), u, v) for u, v in self.G.edges()]
        if not spans:
            return []
        spans.sort(reverse=True)
        max_span = spans[0][0]
        threshold = max(1, max_span - 1)
        return [(u, v, s) for s, u, v in spans if s >= threshold][: self.critical_nodes]

    def _node_features(self):
        n = len(self.nodes)
        pos = self._positions()
        init = {v: i for i, v in enumerate(self.initial_order)}
        bw = max(1, self.current_bw)
        threshold = max(1, self.current_bw - 1)
        feats = []
        for v in self.nodes:
            spans = [abs(pos[v] - pos[u]) for u in self.G.neighbors(v)]
            local_max = max(spans, default=0)
            local_mean = float(np.mean(spans)) if spans else 0.0
            critical_incident = sum(1 for s in spans if s >= threshold)
            degree = max(1, self.G.degree(v))
            deg_norm, core_norm, cluster = self.static[v]
            feats.append([
                deg_norm,
                core_norm,
                cluster,
                pos[v] / max(1, n - 1),
                init[v] / max(1, n - 1),
                local_max / bw,
                local_mean / bw,
                critical_incident / degree,
            ])
        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    def _global_state(self):
        n = max(1, len(self.nodes))
        m = self.G.number_of_edges()
        density = 0.0 if n < 2 else 2.0 * m / (n * (n - 1))
        initial = max(1, self.initial_bw)
        return torch.tensor([
            self.current_bw / initial,
            self.best_bw / initial,
            (self.initial_bw - self.best_bw) / initial,
            self.step_count / self.horizon,
            self.steps_since_improvement / self.horizon,
            min(1.0, math.log1p(n) / 10.0),
            density,
        ], dtype=torch.float32, device=self.device)

    def _build_action_candidates(self):
        n = len(self.order)
        pos = self._positions()
        specs = set()
        for u, v, _ in self._critical_edges():
            iu, iv = pos[u], pos[v]
            for i, target in ((iu, iv), (iv, iu)):
                if i != target:
                    if self.enable_swap:
                        specs.add((0, i, target))
                    if self.enable_relocation:
                        specs.add((1, i, target))
                direction = 1 if target > i else -1
                distance_to_target = abs(target - i)
                for offset in self.offsets:
                    step = min(offset, distance_to_target)
                    j = i + direction * step
                    if 0 <= j < n and j != i:
                        if self.enable_swap:
                            specs.add((0, i, j))
                        if self.enable_relocation:
                            specs.add((1, i, j))
                    for sign in (-1, 1):
                        j = i + sign * offset
                        if 0 <= j < n and j != i:
                            if self.enable_swap:
                                specs.add((0, i, j))
                            if self.enable_relocation:
                                specs.add((1, i, j))
        if not specs and n >= 2:
            specs.add((0, 0, 1))

        self.action_specs = sorted(specs)
        denom = max(1, n - 1)
        rows = []
        for kind, i, j in self.action_specs:
            rows.append([
                float(kind),
                float(i),
                float(j),
                abs(j - i) / denom,
                1.0 if j > i else -1.0,
            ])
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def state(self):
        x = self._node_features()
        node_to_idx = {v: i for i, v in enumerate(self.nodes)}
        order_idx = torch.tensor(
            [node_to_idx[v] for v in self.order],
            dtype=torch.long,
            device=self.device,
        )
        actions = self._build_action_candidates()
        return x, order_idx, actions, self._global_state()

    def step(self, action_index):
        if len(self.order) < 2:
            return self.state(), 0.0, True

        action_index = int(max(0, min(action_index, len(self.action_specs) - 1)))
        kind, i, j = self.action_specs[action_index]
        old_bw = self.current_bw
        old_best = self.best_bw

        if kind == 0:
            self.order[i], self.order[j] = self.order[j], self.order[i]
        else:
            node = self.order.pop(i)
            self.order.insert(j, node)

        self.current_bw = canonical_bandwidth(self.G, self.order)
        self.step_count += 1

        if self.current_bw < self.best_bw:
            self.best_bw = self.current_bw
            self.best_order = list(self.order)
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

        initial = max(1, self.initial_bw)
        immediate = (old_bw - self.current_bw) / initial
        incumbent = (old_best - self.best_bw) / initial
        reward = immediate + 2.0 * incumbent - 0.001

        done = self.step_count >= self.horizon
        return self.state(), float(reward), done


class PPOTrainerV2:
    def __init__(
        self,
        G,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip=0.2,
        epochs=4,
        entropy_coef=0.01,
        horizon=40,
        device=None,
        critical_nodes=24,
    ):
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.env = BandwidthEnvV2(
            G, self.device, horizon=horizon, critical_nodes=critical_nodes
        )
        self.model = ActorCriticV2().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef

    def collect(self, steps):
        transitions = []
        x, order_idx, actions, global_state = self.env.reset()
        steps = min(int(steps), self.env.horizon)

        for _ in range(steps):
            with torch.no_grad():
                logits, value = self.model(
                    x, self.env.edge_index, order_idx, actions, global_state
                )
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            (nx_, no_, na_, ng_), reward, done = self.env.step(action.item())
            transitions.append(Transition(
                x=x.detach(),
                order_idx=order_idx.detach(),
                actions=actions.detach(),
                global_state=global_state.detach(),
                action=int(action.item()),
                logp=float(logp.item()),
                reward=float(reward),
                done=bool(done),
                value=float(value.item()),
            ))
            x, order_idx, actions, global_state = nx_, no_, na_, ng_
            if done:
                break

        return transitions, self.env.best_bw, self.env.best_order

    def _advantages(self, transitions):
        values = [t.value for t in transitions] + [0.0]
        advantages = [0.0] * len(transitions)
        gae = 0.0
        for i in reversed(range(len(transitions))):
            mask = 0.0 if transitions[i].done else 1.0
            delta = transitions[i].reward + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[i] = gae
        returns = [advantages[i] + values[i] for i in range(len(transitions))]
        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if len(adv) > 1 and float(adv.std()) > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return adv, ret

    def update(self, transitions):
        if not transitions:
            return
        advantages, returns = self._advantages(transitions)
        old_logp = torch.tensor(
            [t.logp for t in transitions], dtype=torch.float32, device=self.device
        )

        for _ in range(self.epochs):
            losses = []
            for i, t in enumerate(transitions):
                logits, value = self.model(
                    t.x, self.env.edge_index, t.order_idx, t.actions, t.global_state
                )
                dist = Categorical(logits=logits)
                action = torch.tensor(t.action, device=self.device)
                logp = dist.log_prob(action)
                ratio = torch.exp(logp - old_logp[i])
                s1 = ratio * advantages[i]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[i]
                actor_loss = -torch.min(s1, s2)
                critic_loss = 0.5 * (value - returns[i]).pow(2)
                losses.append(
                    actor_loss + critic_loss - self.entropy_coef * dist.entropy()
                )

            loss = torch.stack(losses).mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

    def train(self, episodes=30, steps=40, seed=123):
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
            history.append({
                "episode": ep + 1,
                "best_bw": int(bw),
                "global_best_bw": int(global_best),
            })

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
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--critical-nodes", type=int, default=24)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    G = load_graph(a.instance)
    trainer = PPOTrainerV2(G, critical_nodes=a.critical_nodes, horizon=a.steps)
    result = trainer.train(a.episodes, a.steps, a.seed)
    result.update({
        "instance": os.path.basename(a.instance),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "version": "gnn-ppo-v2",
    })
    print(json.dumps(result, indent=2))
    if a.output:
        with open(a.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
