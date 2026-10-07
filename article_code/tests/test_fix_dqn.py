import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import networkx as nx

from agent import Agent
import enviroment
from enviroment import Env
from centralities import get_centrality_node, centrality_heuristic
from modules.graph.Grafo import GrafoListaAdj
from modules.utils import read_Instances
from modules.utils.handle_labels import Bf_graph, set_bandwidth


class TestBandwidthFixes(unittest.TestCase):
    def test_classical_bandwidth_path_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3, 4])
        graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        self.assertEqual(set_bandwidth(graph), 1)

    def test_classical_bandwidth_nontrivial_order(self):
        graph = nx.Graph()
        graph.add_nodes_from([1, 3, 2, 4])
        graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        self.assertEqual(set_bandwidth(graph), 2)

    def test_agent_uses_number_of_movements_as_output_size(self):
        agent = Agent(
            learning_rate=0.001,
            gamma=0.9,
            epsilon=0.0,
            eps_min=0.0,
            eps_dec=0.99,
            n_movements=7,
            n_actions=999,
            n_states=4,
            deep=True,
        )
        self.assertEqual(agent.n_actions, 7)
        self.assertEqual(agent.Q.fc3.out_features, 7)

    def test_terminal_dqn_update_runs_without_bootstrap_error(self):
        agent = Agent(
            learning_rate=0.001,
            gamma=0.9,
            epsilon=0.5,
            eps_min=0.1,
            eps_dec=0.9,
            n_movements=3,
            n_actions=3,
            n_states=4,
            deep=True,
        )
        before = agent.epsilon
        agent.learn([0, 0, 0, 10], 1, 1.0, [1, 0, 1, 9], done=True)
        self.assertLess(agent.epsilon, before)

    def test_environment_keeps_solution_matching_best_cost(self):
        calls = [
            (8, "g8", "s8"),
            (7, "g7", "s7"),
            (9, "g9", "s9"),
            (6, "g6", "s6"),
            (10, "g10", "s10"),
        ]

        original = enviroment.centrality_heuristic

        def fake_heuristic(**kwargs):
            cost, graph, solution = calls.pop(0)
            return cost, graph, solution, [{"bandwidth": cost}]

        enviroment.centrality_heuristic = fake_heuristic
        try:
            env = Env(10)
            info = env.step(
                graph=object(),
                centrality_values={},
                cent_str="Degree",
                centralities={},
            )
            self.assertEqual(info["bandwidth"], 6)
            self.assertEqual(info["graph"], "g6")
            self.assertEqual(info["solution"], "s6")
            self.assertEqual(env.best_sol, 6)
            self.assertAlmostEqual(info["reward"], 0.4)
        finally:
            enviroment.centrality_heuristic = original

    def _smoke_matrix(self, filename):
        matrix_path = os.path.join(ROOT, "data", "artigo_datas", filename)
        nnodes, nedges, edges, _, _, _ = read_Instances.load_instance(matrix_path)

        graph_nx = nx.Graph()
        graph_nx.add_nodes_from(range(1, nnodes + 1))
        graph_nx.add_edges_from(edges)

        graph = GrafoListaAdj()
        graph.DefinirN(nnodes, VizinhancaDuplamenteLigada=True)
        for u, v in edges:
            graph.AdicionarAresta(u, v)

        centralities = {
            "Degree": {
                "func": nx.degree_centrality,
                "args": {},
                "reverse": True,
            }
        }
        values = get_centrality_node(graph_nx, centralities["Degree"])
        cost, _, solution, records = centrality_heuristic(
            graph=graph,
            centrality_values=values,
            cent_str="Degree",
            alpha=0.3,
            iter_max=2,
            centralities=centralities,
        )

        self.assertEqual(set(solution.keys()), set(graph.V()))
        self.assertEqual(set(solution.values()), set(range(1, nnodes + 1)))
        self.assertEqual(Bf_graph(graph, solution), cost)
        self.assertGreaterEqual(cost, 0)
        self.assertLessEqual(cost, max(0, nnodes - 1))
        self.assertEqual(len(records), 2)

    def test_real_matrix_bcsstk01(self):
        self._smoke_matrix("bcsstk01.mtx")

    def test_real_matrix_bcsstk02(self):
        self._smoke_matrix("bcsstk02.mtx")


if __name__ == "__main__":
    unittest.main()
