from __future__ import annotations

import os
import sys
import unittest
from unittest import mock


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

    def test_query_gpus_skips_non_ascii_decimal_integer_rows(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        import core.gpu_topology as gpu_topology

        output = "\n".join([
            "0, 24000, 24576",
            "+123, 24000, 24576",
            "1, 1.5, 24576",
            "2, 24000, 1e3",
            "3, -1, 24576",
            "4, 12000, 16384",
        ])
        with mock.patch.object(gpu_topology, "_run", return_value=output):
            self.assertEqual(
                gpu_topology.query_gpus(),
                [
                    gpu_topology.GpuInfo(index=0, mem_free_mib=24000, mem_total_mib=24576),
                    gpu_topology.GpuInfo(index=4, mem_free_mib=12000, mem_total_mib=16384),
                ],
            )


if __name__ == "__main__":
    unittest.main()
