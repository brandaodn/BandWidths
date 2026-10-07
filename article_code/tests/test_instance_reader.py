import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.utils import read_Instances


class TestInstanceReader(unittest.TestCase):
    def test_matrix_market_three_value_header(self):
        p=os.path.join(ROOT,"data","artigo_datas","bcsstk05.mtx")
        n,m,edges,neigh,adj,_=read_Instances.load_instance(p)
        self.assertEqual(n,153)
        self.assertGreater(m,0)
        self.assertEqual(len(adj),153)
        self.assertTrue(all(1 <= u <= n and 1 <= v <= n for u,v in edges))

    def test_legacy_two_value_header(self):
        p=os.path.join(ROOT,"data","artigo_datas","bp_0.mtx")
        n,m,edges,neigh,adj,_=read_Instances.load_instance(p)
        self.assertEqual(n,822)
        self.assertGreater(m,0)
        self.assertEqual(len(adj),822)
        self.assertTrue(all(1 <= u <= n and 1 <= v <= n for u,v in edges))

    def test_edges_are_deduplicated(self):
        p=os.path.join(ROOT,"data","artigo_datas","bcsstk05.mtx")
        _,m,edges,_,_,_=read_Instances.load_instance(p)
        self.assertEqual(m,len(set(edges)))


if __name__=="__main__":
    unittest.main()
