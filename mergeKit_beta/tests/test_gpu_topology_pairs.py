from __future__ import annotations

import os
import sys
import unittest


class TestGpuTopologyPairs(unittest.TestCase):
    def test_parse_default(self):
        # tests 可能从 workspace 根运行，需显式加入 mergeKit_beta 到 sys.path
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from core.gpu_topology import parse_nvlink_pairs

        self.assertEqual(parse_nvlink_pairs(None), [(0, 1), (2, 3)])

    def test_parse_compact(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from core.gpu_topology import parse_nvlink_pairs

        self.assertEqual(parse_nvlink_pairs("01,23"), [(0, 1), (2, 3)])

    def test_parse_dash(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from core.gpu_topology import parse_nvlink_pairs

        self.assertEqual(parse_nvlink_pairs("0-1, 2-3"), [(0, 1), (2, 3)])

    def test_parse_invalid(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from core.gpu_topology import parse_nvlink_pairs

        with self.assertRaises(ValueError):
            parse_nvlink_pairs("2,3")


if __name__ == "__main__":
    unittest.main()
