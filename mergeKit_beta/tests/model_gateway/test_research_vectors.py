import os
import shutil
import tempfile
import unittest

import numpy as np


class TestResearchVectors(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="research_vectors_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_saved_vectors_round_trip_with_the_expected_chunk_count(self):
        from app.model_gateway.vectors import load_chunk_vectors, save_chunk_vectors

        expected = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        path = save_chunk_vectors(self.root, "file-1", expected)

        self.assertTrue(os.path.isfile(path))
        np.testing.assert_array_equal(load_chunk_vectors(self.root, "file-1", expected_rows=2), expected)
        self.assertIsNone(load_chunk_vectors(self.root, "file-1", expected_rows=3))


if __name__ == "__main__":
    unittest.main()
