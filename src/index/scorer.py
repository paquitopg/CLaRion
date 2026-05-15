"""
Cosine-similarity scoring over the document index.

Three backends exposed for the benchmark report:

  * `cosine_python_loop` : pure-Python triple loop. Pedagogical worst case;
                           shows what you have to beat.
  * `cosine_numpy`       : single-call numpy matmul. Implicit BLAS threading.
  * `cosine_cython_omp`  : hand-rolled Cython kernel with OpenMP `prange` over
                           the corpus axis (N).

All three normalize queries and the index to unit length, then compute
cos(q, M_i) = q · M_i (no division needed once both are unit).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("clarion.index.scorer")

try:
    from src.parallel import cython_index as _cy  # type: ignore
    _CY_AVAILABLE = True
except Exception as e:  # pragma: no cover  - extension may not be built yet
    _cy = None
    _CY_AVAILABLE = False
    logger.warning("cython_index extension unavailable, fallback to numpy: %s", e)


# --------------------------------------------------------------------------- #
# Vector normalization helpers
# --------------------------------------------------------------------------- #
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2-normalize a (..., D) array, output is float32, contiguous."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / (norms + eps)).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Backend 1: pure-Python triple loop. Slow on purpose.
# --------------------------------------------------------------------------- #
def cosine_python_loop(queries: np.ndarray, index: np.ndarray) -> np.ndarray:
    """
    Triple-nested loop. O(Q * N * D) in Python overhead.

    Used in benchmarks as the 'how bad does it get without any vectorization'
    reference. Don't call this on N > a few thousand.
    """
    Q, D = queries.shape
    N, D2 = index.shape
    assert D == D2

    # Normalize first (still nested loops, in Python, to keep this honest).
    q_norms = [math.sqrt(sum(float(queries[q, d]) ** 2 for d in range(D))) + 1e-12
               for q in range(Q)]
    i_norms = [math.sqrt(sum(float(index[n, d]) ** 2 for d in range(D))) + 1e-12
               for n in range(N)]

    scores = np.empty((Q, N), dtype=np.float32)
    for q in range(Q):
        for n in range(N):
            dot = 0.0
            for d in range(D):
                dot += float(queries[q, d]) * float(index[n, d])
            scores[q, n] = dot / (q_norms[q] * i_norms[n])
    return scores


# --------------------------------------------------------------------------- #
# Backend 2: numpy / BLAS
# --------------------------------------------------------------------------- #
def cosine_numpy(queries: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Single matmul on L2-normalized inputs. The pragmatic baseline."""
    q = l2_normalize(queries)
    m = l2_normalize(index)
    # Float32 SGEMM via BLAS.
    return q @ m.T


# --------------------------------------------------------------------------- #
# Backend 3: Cython + OpenMP
# --------------------------------------------------------------------------- #
def cosine_cython_omp(
    queries: np.ndarray,
    index: np.ndarray,
    num_threads: int = 0,
) -> np.ndarray:
    """
    Hand-parallel cosine. Falls back to numpy if the extension isn't built.

    The kernel parallelizes the outer N loop (the corpus axis), which is the
    only dimension that grows with the index size. Inner D loop is a tight
    SIMD-friendly accumulator.
    """
    if not _CY_AVAILABLE:
        return cosine_numpy(queries, index)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    m = np.ascontiguousarray(index, dtype=np.float32)
    return _cy.cosine_scores_omp(q, m, num_threads)


# --------------------------------------------------------------------------- #
# Top-k from a score matrix
# --------------------------------------------------------------------------- #
def top_k_indices(
    scores: np.ndarray,
    k: int,
    backend: str = "numpy",
    num_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the top-k indices per query, descending by score.

    Returns (idx, scores) of shape (Q, k) each. The numpy backend uses
    argpartition + a small sort; the Cython backend pushes the loop into C and
    parallelizes the outer query axis.
    """
    Q, N = scores.shape
    k = min(k, N)

    if backend == "cython" and _CY_AVAILABLE:
        return _cy.top_k_descending(np.ascontiguousarray(scores, dtype=np.float32),
                                    k, num_threads)

    # Numpy reference: argpartition is O(N), then sort the k chosen entries.
    part = np.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
    # Gather then sort.
    chosen = np.take_along_axis(scores, part, axis=-1)
    order = np.argsort(-chosen, axis=-1)
    idx = np.take_along_axis(part, order, axis=-1).astype(np.int32, copy=False)
    val = np.take_along_axis(chosen, order, axis=-1).astype(np.float32, copy=False)
    return idx, val


# --------------------------------------------------------------------------- #
# High-level retriever
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalResult:
    indices: np.ndarray  # (Q, k) int32
    scores: np.ndarray   # (Q, k) float32
    backend: str
    wall_time_s: float


class Retriever:
    """Wraps an in-memory index bank and exposes a search API."""

    def __init__(self, bank: np.ndarray):
        assert bank.ndim == 2 and bank.dtype == np.float32
        self.bank = np.ascontiguousarray(bank)
        self._normalized = l2_normalize(self.bank)

    def search(
        self,
        queries: np.ndarray,
        k: int,
        backend: str = "cython",
        num_threads: int = 0,
    ) -> RetrievalResult:
        import time
        t0 = time.perf_counter()

        if backend == "python":
            scores = cosine_python_loop(queries, self.bank)
        elif backend == "numpy":
            scores = l2_normalize(queries) @ self._normalized.T
        elif backend == "cython":
            scores = cosine_cython_omp(queries, self.bank, num_threads)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        idx, vals = top_k_indices(scores, k,
                                  backend="cython" if backend == "cython" else "numpy",
                                  num_threads=num_threads)
        wall = time.perf_counter() - t0
        return RetrievalResult(idx, vals, backend, wall)
