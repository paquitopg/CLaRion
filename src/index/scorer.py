"""
Cosine-similarity scoring over the document index.

CLaRa/CLaRiON-compatible retrieval layer.

Supports:
    - flattened retrieval vectors      : (N, D)
    - memory-token banks               : (N, l, H)
    - differentiable training pipeline
    - ST top-k retrieval
    - decoder memory conditioning

Three backends exposed for benchmarks:

  * cosine_python_loop : pedagogical worst-case baseline
  * cosine_numpy       : BLAS matmul baseline
  * cosine_cython_omp  : OpenMP-parallel Cython kernel

Important:
    Training-time gradients DO NOT flow through this module.
    End-to-end differentiability is handled in pipeline.search_differentiable()
    with torch-native cosine similarity.

This module is inference/index focused.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("clarion.index.scorer")

try:
    from src.parallel import cython_index as _cy  # type: ignore

    _CY_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _cy = None
    _CY_AVAILABLE = False
    logger.warning(
        "cython_index extension unavailable, fallback to numpy: %s",
        e,
    )


# --------------------------------------------------------------------------- #
# Memory-bank helpers
# --------------------------------------------------------------------------- #
def flatten_memory_bank(bank: np.ndarray) -> np.ndarray:
    """
    Flatten a memory-token bank.

    Accepts:
        (N, D)
        (N, l, H)

    Returns:
        (N, D_flat)
    """
    bank = np.ascontiguousarray(bank, dtype=np.float32)

    if bank.ndim == 2:
        return bank

    if bank.ndim != 3:
        raise ValueError(
            f"Expected bank.ndim in {{2,3}}, got {bank.ndim}"
        )

    N, l, H = bank.shape
    return bank.reshape(N, l * H)


def unflatten_memory_vectors(
    vectors: np.ndarray,
    n_memory_tokens: int,
    hidden_dim: int,
) -> np.ndarray:
    """
    Convert flattened vectors back into memory-token tensors.

    Input:
        (B, D_flat)

    Output:
        (B, l, H)
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    return vectors.reshape(
        vectors.shape[0],
        n_memory_tokens,
        hidden_dim,
    )


# --------------------------------------------------------------------------- #
# Vector normalization
# --------------------------------------------------------------------------- #
def l2_normalize(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Row-wise L2 normalization.

    Input:
        (..., D)

    Output:
        (..., D)
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / (norms + eps)).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Backend 1: pure Python
# --------------------------------------------------------------------------- #
def cosine_python_loop(
    queries: np.ndarray,
    index: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Triple nested loop.

    Horrifically slow on purpose.
    """
    queries = flatten_memory_bank(queries)
    index = flatten_memory_bank(index)

    Q, D = queries.shape
    N, D2 = index.shape

    if D != D2:
        raise ValueError(f"Dimension mismatch: {D} != {D2}")

    q_norms = [
        math.sqrt(
            sum(float(queries[q, d]) ** 2 for d in range(D))
        ) + 1e-12
        for q in range(Q)
    ]

    i_norms = [
        math.sqrt(
            sum(float(index[n, d]) ** 2 for d in range(D))
        ) + 1e-12
        for n in range(N)
    ]

    scores = np.empty((Q, N), dtype=np.float32)

    for q in range(Q):
        for n in range(N):
            dot = 0.0

            for d in range(D):
                dot += (
                    float(queries[q, d]) *
                    float(index[n, d])
                )

            scores[q, n] = (
                dot /
                (q_norms[q] * i_norms[n] * temperature)
            )

    return scores


# --------------------------------------------------------------------------- #
# Backend 2: numpy / BLAS
# --------------------------------------------------------------------------- #
def cosine_numpy(
    queries: np.ndarray,
    index: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    BLAS cosine similarity.

    Supports:
        queries : (Q, D)
        index   : (N, D) or (N, l, H)
    """
    q = flatten_memory_bank(queries)
    m = flatten_memory_bank(index)

    q = l2_normalize(q)
    m = l2_normalize(m)

    return (q @ m.T) / temperature


# --------------------------------------------------------------------------- #
# Backend 3: Cython + OpenMP
# --------------------------------------------------------------------------- #
def cosine_cython_omp(
    queries: np.ndarray,
    index: np.ndarray,
    num_threads: int = 0,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    OpenMP-parallel cosine kernel.

    Falls back to numpy if extension unavailable.
    """
    q = flatten_memory_bank(queries)
    m = flatten_memory_bank(index)

    if not _CY_AVAILABLE:
        return cosine_numpy(
            q,
            m,
            temperature=temperature,
        )

    q = np.ascontiguousarray(q, dtype=np.float32)
    m = np.ascontiguousarray(m, dtype=np.float32)

    scores = _cy.cosine_scores_omp(
        q,
        m,
        num_threads,
    )

    return scores / temperature


# --------------------------------------------------------------------------- #
# Top-k
# --------------------------------------------------------------------------- #
def top_k_indices(
    scores: np.ndarray,
    k: int,
    backend: str = "numpy",
    num_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select top-k scores per query.

    Returns:
        idx  : (Q, k)
        vals : (Q, k)
    """
    Q, N = scores.shape
    k = min(k, N)

    if backend == "cython" and _CY_AVAILABLE:
        return _cy.top_k_descending(
            np.ascontiguousarray(scores, dtype=np.float32),
            k,
            num_threads,
        )

    part = np.argpartition(
        -scores,
        kth=k - 1,
        axis=-1,
    )[:, :k]

    chosen = np.take_along_axis(
        scores,
        part,
        axis=-1,
    )

    order = np.argsort(
        -chosen,
        axis=-1,
    )

    idx = np.take_along_axis(
        part,
        order,
        axis=-1,
    ).astype(np.int32, copy=False)

    vals = np.take_along_axis(
        chosen,
        order,
        axis=-1,
    ).astype(np.float32, copy=False)

    return idx, vals


# --------------------------------------------------------------------------- #
# Retrieval result
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalResult:
    """
    Output of retrieval search.

    memory_vectors:
        (Q, k, l, H)

    flat_vectors:
        (Q, k, D_flat)
    """

    indices: np.ndarray
    scores: np.ndarray

    memory_vectors: np.ndarray
    flat_vectors: np.ndarray

    backend: str
    wall_time_s: float

    full_scores: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """
    In-memory retrieval wrapper.

    Stores:
        - flattened retrieval bank
        - structured memory-token bank

    Compatible with:
        - inference retrieval
        - differentiable ST top-k pipeline
        - decoder memory conditioning
    """

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

            self.bank_memory = bank.reshape(
                bank.shape[0],
                n_memory_tokens,
                hidden_dim,
            )

        elif bank.ndim == 3:
            self.bank_memory = bank
            self.bank_flat = flatten_memory_bank(bank)

        else:
            raise ValueError(
                f"Expected bank.ndim in {{2,3}}, got {bank.ndim}"
            )

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
        """
        Retrieve top-k memory entries.

        Queries are expected to already be flattened:
            (Q, D_flat)
        """
        t0 = time.perf_counter()

        queries = flatten_memory_bank(queries)

        if backend == "python":
            scores = cosine_python_loop(
                queries,
                self.bank_flat,
                temperature=temperature,
            )

        elif backend == "numpy":
            q = l2_normalize(queries)

            scores = (
                q @ self._normalized.T
            ) / temperature

        elif backend == "cython":
            scores = cosine_cython_omp(
                queries,
                self.bank_flat,
                num_threads=num_threads,
                temperature=temperature,
            )

        else:
            raise ValueError(
                f"Unknown backend: {backend!r}"
            )

        idx, vals = top_k_indices(
            scores,
            k,
            backend=(
                "cython"
                if backend == "cython"
                else "numpy"
            ),
            num_threads=num_threads,
        )

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

            full_scores=(
                scores
                if return_scores_matrix
                else None
            ),
        )