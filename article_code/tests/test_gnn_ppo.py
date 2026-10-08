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

    def test_expanded_action_space_contains_swap_and_relocation(self):
        import torch
        G=nx.path_graph(range(1,41))
        env=BandwidthEnv(G,device=torch.device("cpu"))
        env.reset()
        kinds={kind for kind,_,_ in env.action_specs}
        distances={abs(i-j) for _,i,j in env.action_specs}
        self.assertIn(0,kinds)
        self.assertIn(1,kinds)
        self.assertTrue(any(distance > 1 for distance in distances))

    def test_critical_edge_guided_actions_exist(self):
        import torch
        G=nx.Graph()
        G.add_nodes_from(range(1,9))
        G.add_edges_from([(1,8),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)])
        env=BandwidthEnv(G,device=torch.device("cpu"))
        env.order=[1,2,3,4,5,6,7,8]
        env.current_bw=canonical_bandwidth(G,env.order)
        env._build_action_candidates()
        pos={v:i for i,v in enumerate(env.order)}
        i,j=pos[1],pos[8]
        direct={(kind,a,b) for kind,a,b in env.action_specs}
        self.assertTrue((0,i,j) in direct or (1,i,j) in direct)

    def test_ppo_smoke(self):
        G=nx.cycle_graph(range(1,9))
        trainer=PPOTrainer(G,device=__import__("torch").device("cpu"),epochs=1)
        result=trainer.train(episodes=2,steps=5,seed=7)
        self.assertLessEqual(result["best_bw"],result["rcm_bw"])
        self.assertEqual(len(result["history"]),2)


if __name__=="__main__":
    unittest.main()
