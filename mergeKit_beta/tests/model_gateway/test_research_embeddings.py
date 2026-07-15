import unittest

import numpy as np


class _FakeInput:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    def __init__(self):
        self.providers = None
        self.feed = None

    def get_inputs(self):
        return [_FakeInput("input_ids"), _FakeInput("attention_mask")]

    def run(self, _outputs, feed):
        self.feed = feed
        # The BGE-M3 ONNX model returns token embeddings; CLS pooling is configured
        # by its checked-in SentenceTransformers pooling module.
        return [np.array([
            [[3.0] + [0.0] * 1023],
            [[0.0, 4.0] + [0.0] * 1022],
        ], dtype=np.float32)]


class _FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        return {
            "input_ids": np.array([[11, 12], [21, 22]], dtype=np.int64),
            "attention_mask": np.ones((len(texts), 2), dtype=np.int64),
        }


class TestOnnxDenseEmbedder(unittest.TestCase):
    def test_cpu_onnx_encoder_uses_cls_pooling_and_l2_normalizes(self):
        from app.model_gateway.embeddings import OnnxDenseEmbedder

        session = _FakeSession()
        embedder = OnnxDenseEmbedder(
            "/models/bge-m3",
            session_factory=lambda path, providers: self._record_session(session, path, providers),
            tokenizer_factory=lambda path: _FakeTokenizer(),
        )

        vectors = embedder.encode(["alpha", "beta"])

        self.assertEqual(session.providers, ["CPUExecutionProvider"])
        self.assertEqual(set(session.feed), {"input_ids", "attention_mask"})
        self.assertEqual(vectors.shape, (2, 1024))
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0]))
        self.assertTrue(np.allclose(vectors[0, :2], [1.0, 0.0]))
        self.assertTrue(np.allclose(vectors[1, :2], [0.0, 1.0]))

    @staticmethod
    def _record_session(session, path, providers):
        session.path = path
        session.providers = providers
        return session


if __name__ == "__main__":
    unittest.main()
