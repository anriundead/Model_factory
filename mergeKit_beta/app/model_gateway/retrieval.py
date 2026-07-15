"""Short-lived, owner-scoped hybrid retrieval for research citations."""
from __future__ import annotations

from datetime import datetime
import re

import faiss
import numpy as np

from app.model_gateway.models import ResearchChunk
from app.model_gateway.vectors import load_chunk_vectors


def _lexical_score(query: str, text: str) -> float:
    query_terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", query.lower()))
    if not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)


def retrieve_research_evidence(
    session, encoder, api_key_id: str, file_ids: list[str], query: str, limit: int = 8, *, runtime_root: str | None = None,
) -> list[dict]:
    """Return citation-ready evidence without retaining vectors or crossing key scope."""
    requested_files = [str(file_id) for file_id in file_ids if file_id]
    if not requested_files or not (query or "").strip():
        return []
    if limit < 1:
        raise ValueError("retrieval limit must be positive")

    chunks = (
        session.query(ResearchChunk)
        .filter(
            ResearchChunk.api_key_id == api_key_id,
            ResearchChunk.file_id.in_(requested_files),
            ResearchChunk.expires_at > datetime.utcnow(),
        )
        .order_by(ResearchChunk.file_id.asc(), ResearchChunk.ordinal.asc())
        .all()
    )
    if not chunks:
        return []

    chunk_vectors = None
    if runtime_root:
        grouped = {}
        for chunk in chunks:
            grouped.setdefault(chunk.file_id, []).append(chunk)
        vector_groups = []
        for file_id in sorted(grouped):
            vectors = load_chunk_vectors(runtime_root, file_id, expected_rows=len(grouped[file_id]))
            if vectors is None:
                break
            vector_groups.append(vectors)
        else:
            chunk_vectors = np.concatenate(vector_groups, axis=0)

    if chunk_vectors is None and runtime_root:
        ranked = sorted(
            ((0.0, 0.0, _lexical_score(query, chunk.text), chunk) for chunk in chunks),
            key=lambda item: (-item[2], item[3].file_id, item[3].ordinal),
        )
        return [
            {
                "chunk_id": item[3].id,
                "file_id": item[3].file_id,
                "text": item[3].text,
                "locator": item[3].locator,
                "semantic_score": item[1],
                "lexical_score": round(item[2], 6),
            }
            for item in ranked[:limit]
        ]
    if chunk_vectors is None:
        vectors = np.asarray(encoder.encode([query] + [chunk.text for chunk in chunks]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) + 1:
            raise RuntimeError("research embedder returned an invalid vector batch")
        query_vector, chunk_vectors = vectors[:1], vectors[1:]
    else:
        query_vector = np.asarray(encoder.encode([query]), dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1 or query_vector.shape[1] != chunk_vectors.shape[1]:
            raise RuntimeError("research query embedding has an invalid shape")
    index = faiss.IndexFlatIP(chunk_vectors.shape[1])
    index.add(np.ascontiguousarray(chunk_vectors))
    candidate_count = min(len(chunks), max(limit * 4, limit))
    semantic_scores, candidate_indexes = index.search(np.ascontiguousarray(query_vector), candidate_count)

    ranked = []
    for semantic, chunk_index in zip(semantic_scores[0], candidate_indexes[0]):
        if chunk_index < 0:
            continue
        chunk = chunks[int(chunk_index)]
        lexical = _lexical_score(query, chunk.text)
        ranked.append((float(semantic) * 0.85 + lexical * 0.15, float(semantic), lexical, chunk))
    ranked.sort(key=lambda item: (-item[0], item[3].file_id, item[3].ordinal))
    return [
        {
            "chunk_id": item[3].id,
            "file_id": item[3].file_id,
            "text": item[3].text,
            "locator": item[3].locator,
            "semantic_score": round(item[1], 6),
            "lexical_score": round(item[2], 6),
        }
        for item in ranked[:limit]
    ]
