import os
import sys
import unittest

import networkx as nx
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gnn_ppo_v2 import BandwidthEnvV2, PPOTrainerV2, ActorCriticV2


class TestGNNPPOV2(unittest.TestCase):
    def setUp(self):
        self.G = nx.path_graph(range(1, 9))
        self.device = torch.device("cpu")

    def test_state_shapes_and_finiteness(self):
        env = BandwidthEnvV2(self.G, self.device, horizon=8)
        x, order_idx, actions, global_state = env.state()
        self.assertEqual(x.shape, (8, 8))
        self.assertEqual(order_idx.shape, (8,))
        self.assertEqual(global_state.shape, (7,))
        self.assertGreater(actions.shape[0], 0)
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(torch.isfinite(global_state).all())

    def test_sparse_actor_critic_forward(self):
        env = BandwidthEnvV2(self.G, self.device, horizon=8)
        x, order_idx, actions, global_state = env.state()
        model = ActorCriticV2()
        logits, value = model(
            x, env.edge_index, order_idx, actions, global_state
        )
        self.assertEqual(logits.shape[0], actions.shape[0])
        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value))

    def test_smoke_train(self):
        trainer = PPOTrainerV2(
            self.G,
            device=self.device,
            horizon=6,
            critical_nodes=4,
            epochs=1,
        )
        result = trainer.train(episodes=2, steps=6, seed=7)
        self.assertIn("rcm_bw", result)
        self.assertIn("best_bw", result)
        self.assertLessEqual(result["best_bw"], result["rcm_bw"])
        self.assertEqual(len(result["history"]), 2)


if __name__ == "__main__":
    unittest.main()
