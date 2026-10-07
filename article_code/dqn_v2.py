import math
from dataclasses import dataclass, field
import numpy as np

from agent_v2 import DQNAgentV2, DQNConfig
from centralities import centrality_heuristic


@dataclass
class SearchState:
    initial_bw: float
    n_actions: int
    horizon: int
    step: int = 0
    best_bw: float = None
    last_bw: float = None
    steps_since_improvement: int = 0
    attempts: np.ndarray = field(init=False)
    successes: np.ndarray = field(init=False)

    def __post_init__(self):
        self.best_bw = float(self.initial_bw)
        self.last_bw = float(self.initial_bw)
        self.attempts = np.zeros(self.n_actions, dtype=np.float32)
        self.successes = np.zeros(self.n_actions, dtype=np.float32)

    def vector(self):
        denom = max(1.0, abs(self.initial_bw))
        success_rates = np.divide(
            self.successes,
            np.maximum(self.attempts, 1.0),
            dtype=np.float32,
        )
        improvement = (self.initial_bw - self.best_bw) / denom
        return np.asarray([
            self.step / max(1, self.horizon),
            self.last_bw / denom,
            self.best_bw / denom,
            improvement,
            self.steps_since_improvement / max(1, self.horizon),
            *success_rates.tolist(),
        ], dtype=np.float32)


def dense_reward(candidate_bw, previous_best, initial_bw):
    """Dense reward: positive improvement, small tie bonus, mild penalty otherwise."""
    denom = max(1.0, abs(initial_bw))
    delta = (previous_best - candidate_bw) / denom
    if delta > 0:
        return float(delta)
    if candidate_bw == previous_best:
        return 0.01
    return float(-0.05 * min((candidate_bw - previous_best) / denom, 1.0))


class DQNSearchV2:
    def __init__(self, graph, centrality_maps, centralities, initial_bw,
                 train_steps=500, eval_steps=100, samples_per_action=15):
        self.graph = graph
        self.centrality_maps = centrality_maps
        self.centralities = centralities
        self.names = list(centralities.keys())
        self.initial_bw = float(initial_bw)
        self.train_steps = int(train_steps)
        self.eval_steps = int(eval_steps)
        self.samples_per_action = int(samples_per_action)

        state_dim = 5 + len(self.names)
        cfg = DQNConfig(
            state_dim=state_dim,
            n_actions=len(self.names),
            horizon=self.train_steps,
            epsilon_start=1.0,
            epsilon_end=0.01,
            replay_capacity=max(10000, self.train_steps * 4),
            batch_size=64,
            warmup=64,
            target_update_interval=25,
        )
        self.agent = DQNAgentV2(cfg)

    def evaluate_action(self, name):
        # Preserve the existing constructive heuristic. One action receives exactly
        # samples_per_action randomized constructions.
        cost, _, _, _ = centrality_heuristic(
            graph=self.graph,
            centrality_values=self.centrality_maps[name],
            cent_str=name,
            alpha=0.3,
            iter_max=self.samples_per_action,
            centralities=self.centralities,
        )
        return float(cost)

    def _transition(self, state_obj, action, candidate_bw):
        previous_best = state_obj.best_bw
        reward = dense_reward(candidate_bw, previous_best, self.initial_bw)
        improved = candidate_bw < previous_best

        state_obj.attempts[action] += 1
        if improved:
            state_obj.successes[action] += 1
            state_obj.best_bw = candidate_bw
            state_obj.steps_since_improvement = 0
        else:
            state_obj.steps_since_improvement += 1

        state_obj.last_bw = candidate_bw
        state_obj.step += 1
        return reward, improved

    def train(self):
        s = SearchState(self.initial_bw, len(self.names), self.train_steps)
        history = []
        for i in range(self.train_steps):
            state = s.vector()
            epsilon_before = self.agent.epsilon
            action = self.agent.choose_action(state, explore=True)
            name = self.names[action]
            candidate = self.evaluate_action(name)
            reward, improved = self._transition(s, action, candidate)
            next_state = s.vector()
            done = i == self.train_steps - 1

            self.agent.remember(state, action, reward, next_state, done)
            loss = self.agent.learn()

            history.append({
                "phase": "train",
                "step": i + 1,
                "epsilon": epsilon_before,
                "action": action,
                "centrality": name,
                "candidate_bw": candidate,
                "best_bw": s.best_bw,
                "reward": reward,
                "improved": int(improved),
                "loss": loss,
            })
        return s, history

    def evaluate_greedy(self):
        # Fresh search state, frozen network, epsilon=0 behavior.
        s = SearchState(self.initial_bw, len(self.names), max(1, self.eval_steps))
        history = []
        for i in range(self.eval_steps):
            state = s.vector()
            action = self.agent.choose_action(state, explore=False)
            name = self.names[action]
            candidate = self.evaluate_action(name)
            reward, improved = self._transition(s, action, candidate)
            history.append({
                "phase": "eval",
                "step": i + 1,
                "epsilon": 0.0,
                "action": action,
                "centrality": name,
                "candidate_bw": candidate,
                "best_bw": s.best_bw,
                "reward": reward,
                "improved": int(improved),
                "loss": None,
            })
        return s, history
