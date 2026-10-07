import os
import sys
import unittest

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)

import networkx as nx
from gnn_ppo import BandwidthEnv,PPOTrainer,canonical_bandwidth,rcm_order


class TestGNNPPO(unittest.TestCase):
    def test_bandwidth_path(self):
        G=nx.path_graph([1,2,3,4])
        order=[1,2,3,4]
        self.assertEqual(canonical_bandwidth(G,order),1)

    def test_rcm_initialization(self):
        G=nx.path_graph([1,2,3,4,5])
        order=rcm_order(G)
        self.assertEqual(canonical_bandwidth(G,order),1)

    def test_environment_preserves_best(self):
        G=nx.path_graph([1,2,3,4,5])
        env=BandwidthEnv(G,device=__import__("torch").device("cpu"))
        start=env.best_bw
        env.step(1)
        self.assertLessEqual(env.best_bw,start)

    def test_ppo_smoke(self):
        G=nx.cycle_graph(range(1,9))
        trainer=PPOTrainer(G,device=__import__("torch").device("cpu"),epochs=1)
        result=trainer.train(episodes=2,steps=5,seed=7)
        self.assertLessEqual(result["best_bw"],result["rcm_bw"])
        self.assertEqual(len(result["history"]),2)


if __name__=="__main__":
    unittest.main()
