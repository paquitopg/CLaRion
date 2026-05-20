from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("clarion.index.scorer")

try:
    from src.parallel import cython_index as _cy
    _CY_AVAILABLE = True
except Exception as e:
    _cy = None
    _CY_AVAILABLE = False
    logger.warning(
        "cython_index extension unavailable, fallback to numpy: %s",
        e,
    )


def flatten_memory_bank(bank: np.ndarray) -> np.ndarray:
    bank = np.ascontiguousarray(bank, dtype=np.float32)

    if bank.ndim == 2:
        return bank

    if bank.ndim != 3:
        raise ValueError(f"Expected bank.ndim in {{2,3}}, got {bank.ndim}")

    n, l, h = bank.shape
    return bank.reshape(n, l * h)


def unflatten_memory_vectors(
    vectors: np.ndarray,
    n_memory_tokens: int,
    hidden_dim: int,
) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    return vectors.reshape(vectors.shape[0], n_memory_tokens, hidden_dim)


def l2_normalize(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / (norms + eps)).astype(np.float32, copy=False)


def cosine_python_loop(
    queries: np.ndarray,
    index: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    queries = flatten_memory_bank(queries)
    index = flatten_memory_bank(index)

    q, d = queries.shape
    n, d2 = index.shape

    if d != d2:
        raise ValueError(f"Dimension mismatch: {d} != {d2}")

    q_norms = [
        math.sqrt(sum(float(queries[i, j]) ** 2 for j in range(d))) + 1e-12
        for i in range(q)
    ]
    i_norms = [
        math.sqrt(sum(float(index[i, j]) ** 2 for j in range(d))) + 1e-12
        for i in range(n)
    ]

    scores = np.empty((q, n), dtype=np.float32)

    for i in range(q):
        for j in range(n):
            dot = 0.0
            for k in range(d):
                dot += float(queries[i, k]) * float(index[j, k])
            scores[i, j] = dot / (q_norms[i] * i_norms[j] * temperature)

    return scores


def cosine_numpy(
    queries: np.ndarray,
    index: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    q = l2_normalize(flatten_memory_bank(queries))
    m = l2_normalize(flatten_memory_bank(index))
    return (q @ m.T) / temperature


def top_k_indices(
    scores: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    q, n = scores.shape
    k = min(k, n)

    if k <= 0:
        raise ValueError("k must be >= 1")

    part = np.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
    chosen = np.take_along_axis(scores, part, axis=-1)
    order = np.argsort(-chosen, axis=-1)

    idx = np.take_along_axis(part, order, axis=-1).astype(np.int32, copy=False)
    vals = np.take_along_axis(chosen, order, axis=-1).astype(np.float32, copy=False)

    return idx, vals


def cosine_cython_omp(
    queries: np.ndarray,
    index: np.ndarray,
    num_threads: int = 0,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    API conservée pour compatibilité.
    Mais ce chemin ne devrait plus être utilisé pour la recherche top-k.
    """
    q = l2_normalize(flatten_memory_bank(queries))
    m = l2_normalize(flatten_memory_bank(index))

    if not _CY_AVAILABLE or not hasattr(_cy, "cosine_scores_omp"):
        return (q @ m.T) / temperature

    scores = _cy.cosine_scores_omp(
        np.ascontiguousarray(q, dtype=np.float32),
        np.ascontiguousarray(m, dtype=np.float32),
        num_threads,
    )

    if temperature != 1.0:
        scores = scores / temperature

    return scores


def cosine_topk_cython_backend(
    queries: np.ndarray,
    index_normed: np.ndarray,
    k: int,
    num_threads: int = 0,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Backend 'cython' réel = fused top-k.
    Retourne (idx, vals, full_scores_or_none)
    """
    q = l2_normalize(flatten_memory_bank(queries))
    m = np.ascontiguousarray(index_normed, dtype=np.float32)

    if k <= 0:
        raise ValueError("k must be >= 1")

    if _CY_AVAILABLE and hasattr(_cy, "cosine_top_k_omp"):
        idx, vals = _cy.cosine_top_k_omp(
            np.ascontiguousarray(q, dtype=np.float32),
            m,
            k,
            num_threads,
        )
        if temperature != 1.0:
            vals = vals / temperature
        return (
            idx.astype(np.int32, copy=False),
            vals.astype(np.float32, copy=False),
            None,
        )

    scores = q @ m.T
    if temperature != 1.0:
        scores = scores / temperature
    idx, vals = top_k_indices(scores, k)
    return idx, vals, scores


@dataclass
class RetrievalResult:
    indices: np.ndarray
    scores: np.ndarray
    memory_vectors: np.ndarray
    flat_vectors: np.ndarray
    backend: str
    wall_time_s: float
    full_scores: Optional[np.ndarray] = None


class Retriever:
    def __init__(
        self,
        bank: np.ndarray,
        n_memory_tokens: int,
        hidden_dim: int,
    ):
        bank = np.ascontiguousarray(bank, dtype=np.float32)

        self.n_memory_tokens = n_memory_tokens
        self.hidden_dim = hidden_dim

        if bank.ndim == 2:
            self.bank_flat = bank
            self.bank_memory = bank.reshape(bank.shape[0], n_memory_tokens, hidden_dim)
        elif bank.ndim == 3:
            self.bank_memory = bank
            self.bank_flat = flatten_memory_bank(bank)
        else:
            raise ValueError(f"Expected bank.ndim in {{2,3}}, got {bank.ndim}")

        self._normalized = l2_normalize(self.bank_flat)

    def search(
        self,
        queries: np.ndarray,
        k: int,
        backend: str = "cython",
        num_threads: int = 0,
        temperature: float = 1.0,
        return_scores_matrix: bool = False,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        queries = flatten_memory_bank(queries)

        scores = None

        if backend == "python":
            scores = cosine_python_loop(
                queries,
                self.bank_flat,
                temperature=temperature,
            )
            idx, vals = top_k_indices(scores, k)

        elif backend == "numpy":
            q = l2_normalize(queries)
            scores = q @ self._normalized.T
            if temperature != 1.0:
                scores = scores / temperature
            idx, vals = top_k_indices(scores, k)

        elif backend == "cython":
            idx, vals, scores = cosine_topk_cython_backend(
                queries=queries,
                index_normed=self._normalized,
                k=k,
                num_threads=num_threads,
                temperature=temperature,
            )

        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        flat_vectors = self.bank_flat[idx]
        memory_vectors = self.bank_memory[idx]
        wall = time.perf_counter() - t0

        return RetrievalResult(
            indices=idx,
            scores=vals,
            flat_vectors=flat_vectors,
            memory_vectors=memory_vectors,
            backend=backend,
            wall_time_s=wall,
            full_scores=scores if return_scores_matrix else None,
        )