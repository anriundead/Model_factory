"""CPU-only dense embeddings for short-lived research chunks."""
from __future__ import annotations

import os

import numpy as np


EMBEDDING_DIMENSION = 1024
MAX_EMBED_TOKENS = 1024


class OnnxDenseEmbedder:
    """Run the pinned BGE-M3 ONNX export without loading Torch or a GPU."""

    def __init__(self, model_dir: str, session_factory=None, tokenizer_factory=None):
        self.model_dir = os.path.abspath(model_dir)
        onnx_path = os.path.join(self.model_dir, "onnx", "model.onnx")
        if session_factory is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required for research embeddings") from exc

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session_factory = lambda path, providers: ort.InferenceSession(path, options, providers=providers)
        if tokenizer_factory is None:
            from transformers import AutoTokenizer
            tokenizer_factory = lambda path: AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )

        self.session = session_factory(onnx_path, ["CPUExecutionProvider"])
        self.tokenizer = tokenizer_factory(self.model_dir)
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        tokens = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=MAX_EMBED_TOKENS, return_tensors="np"
        )
        feed = {name: value for name, value in tokens.items() if name in self.input_names}
        token_embeddings = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
        if token_embeddings.ndim != 3 or token_embeddings.shape[2] != EMBEDDING_DIMENSION:
            raise RuntimeError("unexpected research embedding model output")
        vectors = token_embeddings[:, 0, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, np.finfo(np.float32).eps)
