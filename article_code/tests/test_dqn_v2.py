import os
import sys
import unittest
import random
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent_v2 import DQNAgentV2, DQNConfig
from dqn_v2 import SearchState, dense_reward


class TestDQNV2(unittest.TestCase):
    def test_epsilon_reaches_target_at_horizon(self):
        cfg = DQNConfig(state_dim=12, n_actions=7, horizon=500)
        agent = DQNAgentV2(cfg)
        agent.env_steps = 500
        self.assertAlmostEqual(agent.epsilon, 0.01, places=6)

    def test_state_is_normalized(self):
        s = SearchState(initial_bw=100, n_actions=7, horizon=500)
        s.step = 250
        s.last_bw = 90
        s.best_bw = 80
        s.steps_since_improvement = 25
        s.attempts[0] = 4
        s.successes[0] = 2
        v = s.vector()
        self.assertEqual(len(v), 12)
        self.assertAlmostEqual(v[0], 0.5)
        self.assertAlmostEqual(v[1], 0.9)
        self.assertAlmostEqual(v[2], 0.8)
        self.assertAlmostEqual(v[3], 0.2)
        self.assertAlmostEqual(v[4], 0.05)
        self.assertAlmostEqual(v[5], 0.5)

    def test_dense_reward_has_signal_for_non_improvement(self):
        self.assertGreater(dense_reward(9, 10, 10), 0)
        self.assertGreater(dense_reward(10, 10, 10), 0)
        self.assertLess(dense_reward(12, 10, 10), 0)

    def test_replay_and_target_learning_run(self):
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        cfg = DQNConfig(
            state_dim=12, n_actions=7, horizon=10,
            replay_capacity=100, batch_size=4, warmup=4,
            target_update_interval=2,
        )
        agent = DQNAgentV2(cfg)
        for i in range(4):
            s = np.zeros(12, dtype=np.float32)
            ns = np.ones(12, dtype=np.float32) * (i + 1) / 10
            agent.remember(s, i % 7, 0.1, ns, i == 3)
        loss = agent.learn()
        self.assertIsInstance(loss, float)
        self.assertEqual(agent.learn_steps, 1)

    def test_greedy_action_does_not_advance_epsilon_schedule(self):
        cfg = DQNConfig(state_dim=12, n_actions=7, horizon=100)
        agent = DQNAgentV2(cfg)
        before = agent.env_steps
        agent.choose_action(np.zeros(12, dtype=np.float32), explore=False)
        self.assertEqual(agent.env_steps, before)


if __name__ == "__main__":
    unittest.main()
