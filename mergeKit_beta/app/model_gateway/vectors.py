"""TTL-bound vector sidecars for short-lived research chunks."""
from __future__ import annotations

import os

import numpy as np


def vector_path(root: str, file_id: str) -> str:
    return os.path.join(os.path.abspath(root), "vectors", f"{file_id}.npy")


def save_chunk_vectors(root: str, file_id: str, vectors) -> str:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("invalid_research_vectors")
    path = vector_path(root, file_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)
    return path


def load_chunk_vectors(root: str, file_id: str, *, expected_rows: int):
    path = vector_path(root, file_id)
    try:
        with open(path, "rb") as handle:
            values = np.load(handle, allow_pickle=False)
    except (OSError, ValueError):
        return None
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != int(expected_rows):
        return None
    return values


def encode_chunk_vectors(encoder, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    batches = [encoder.encode(texts[start:start + batch_size]) for start in range(0, len(texts), batch_size)]
    return np.concatenate([np.asarray(batch, dtype=np.float32) for batch in batches], axis=0)
