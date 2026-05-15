# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: language_level=3
"""
CLaRiON Cython + OpenMP retrieval kernels.

Two public functions:

  cosine_scores_omp(queries, index, num_threads) -> (Q, N) scores matrix
  top_k_descending(scores, k, num_threads)      -> (indices, values), both (Q, k)

The cosine kernel parallelizes the outer corpus axis (N), which is the only
axis that scales with index size. Per-thread work is a single tight dot
product (SIMD-vectorizable under -O3 -march=native). Norms are precomputed
once and reused across queries to amortize sqrt cost.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free

ctypedef np.float32_t F32
ctypedef np.int32_t I32


# --------------------------------------------------------------------------- #
# Cosine similarity: every query against every doc in the index.
# --------------------------------------------------------------------------- #
def cosine_scores_omp(
    np.ndarray[F32, ndim=2, mode="c"] queries,
    np.ndarray[F32, ndim=2, mode="c"] index,
    int num_threads = 0,
):
    """
    Args
    ----
    queries : (Q, D) float32 contiguous
    index   : (N, D) float32 contiguous
    num_threads : OpenMP thread count (0 means use OMP default)

    Returns
    -------
    scores : (Q, N) float32 — cos(queries[q], index[n])
    """
    cdef int Q = queries.shape[0]
    cdef int N = index.shape[0]
    cdef int D = queries.shape[1]
    assert index.shape[1] == D, "queries and index dim mismatch"

    cdef np.ndarray[F32, ndim=2] scores = np.empty((Q, N), dtype=np.float32)
    cdef np.ndarray[F32, ndim=1] qnorms = np.empty(Q, dtype=np.float32)
    cdef np.ndarray[F32, ndim=1] inorms = np.empty(N, dtype=np.float32)

    cdef float* q_ptr = <float*> queries.data
    cdef float* m_ptr = <float*> index.data
    cdef float* s_ptr = <float*> scores.data
    cdef float* qn    = <float*> qnorms.data
    cdef float* mn    = <float*> inorms.data

    cdef int q, n, d
    cdef float acc, qn_v, mn_v
    cdef int nt = num_threads if num_threads > 0 else 0

    # NB: `acc = acc + ...` instead of `acc += ...` so Cython doesn't treat
    # `acc` as an OpenMP reduction variable (which would forbid reading it
    # in the same iteration body).

    # ---- Precompute || queries[q] || ----
    if nt > 0:
        for q in prange(Q, nogil=True, schedule='static', num_threads=nt):
            acc = 0.0
            for d in range(D):
                acc = acc + q_ptr[q * D + d] * q_ptr[q * D + d]
            qn[q] = sqrt(acc) + 1e-12
    else:
        for q in prange(Q, nogil=True, schedule='static'):
            acc = 0.0
            for d in range(D):
                acc = acc + q_ptr[q * D + d] * q_ptr[q * D + d]
            qn[q] = sqrt(acc) + 1e-12

    # ---- Precompute || index[n] || ----
    if nt > 0:
        for n in prange(N, nogil=True, schedule='static', num_threads=nt):
            acc = 0.0
            for d in range(D):
                acc = acc + m_ptr[n * D + d] * m_ptr[n * D + d]
            mn[n] = sqrt(acc) + 1e-12
    else:
        for n in prange(N, nogil=True, schedule='static'):
            acc = 0.0
            for d in range(D):
                acc = acc + m_ptr[n * D + d] * m_ptr[n * D + d]
            mn[n] = sqrt(acc) + 1e-12

    # ---- Main loop: for each query, parallel sweep over the corpus. ----
    # Q is usually small (tens-hundreds), N is large (10k+), so we put
    # `prange` on N and keep `q` in the outer Python loop. This gives every
    # thread a stable cache line of `queries[q]`.
    for q in range(Q):
        qn_v = qn[q]
        if nt > 0:
            for n in prange(N, nogil=True, schedule='static', num_threads=nt):
                acc = 0.0
                for d in range(D):
                    acc = acc + q_ptr[q * D + d] * m_ptr[n * D + d]
                s_ptr[q * N + n] = acc / (qn_v * mn[n])
        else:
            for n in prange(N, nogil=True, schedule='static'):
                acc = 0.0
                for d in range(D):
                    acc = acc + q_ptr[q * D + d] * m_ptr[n * D + d]
                s_ptr[q * N + n] = acc / (qn_v * mn[n])

    return scores


# --------------------------------------------------------------------------- #
# Top-k descending. One query per thread; per-query we keep a small heap.
# Returns sorted indices + values, so the result is downstream-friendly for
# the differentiable top-k aggregator on the decoder side.
# --------------------------------------------------------------------------- #
def top_k_descending(
    np.ndarray[F32, ndim=2, mode="c"] scores,
    int k,
    int num_threads = 0,
):
    """
    Args
    ----
    scores : (Q, N) float32 contiguous
    k      : how many of the top scores to return per query
    num_threads : OpenMP thread count (0 = OMP default)

    Returns
    -------
    indices : (Q, k) int32   — sorted descending by score
    values  : (Q, k) float32 — corresponding scores
    """
    cdef int Q = scores.shape[0]
    cdef int N = scores.shape[1]
    cdef int kk = k if k <= N else N

    cdef np.ndarray[I32, ndim=2] indices = np.empty((Q, kk), dtype=np.int32)
    cdef np.ndarray[F32, ndim=2] values  = np.empty((Q, kk), dtype=np.float32)

    cdef float* s_ptr = <float*> scores.data
    cdef int*   i_ptr = <int*>   indices.data
    cdef float* v_ptr = <float*> values.data

    cdef int q, n, j, m, swap_i
    cdef float swap_v, cand_v, cur_v
    cdef int cand_i
    cdef int nt = num_threads if num_threads > 0 else 0

    # Selection-sort over k is O(k * N), which for k <= ~50 beats a heap in
    # practice because it stays branch-predictable and cache-friendly.
    if nt > 0:
        for q in prange(Q, nogil=True, schedule='static', num_threads=nt):
            # Initialize the top-k buffer with the first kk scores.
            for j in range(kk):
                i_ptr[q * kk + j] = j
                v_ptr[q * kk + j] = s_ptr[q * N + j]

            # Sort the initial buffer descending (small kk).
            for j in range(kk - 1):
                m = j
                for n in range(j + 1, kk):
                    if v_ptr[q * kk + n] > v_ptr[q * kk + m]:
                        m = n
                if m != j:
                    swap_v = v_ptr[q * kk + j]
                    v_ptr[q * kk + j] = v_ptr[q * kk + m]
                    v_ptr[q * kk + m] = swap_v
                    swap_i = i_ptr[q * kk + j]
                    i_ptr[q * kk + j] = i_ptr[q * kk + m]
                    i_ptr[q * kk + m] = swap_i

            # Stream the remaining scores, inserting into the sorted buffer.
            for n in range(kk, N):
                cand_v = s_ptr[q * N + n]
                if cand_v <= v_ptr[q * kk + kk - 1]:
                    continue
                # Replace tail then shift up to keep sorted descending.
                v_ptr[q * kk + kk - 1] = cand_v
                i_ptr[q * kk + kk - 1] = n
                for j in range(kk - 1, 0, -1):
                    if v_ptr[q * kk + j] > v_ptr[q * kk + j - 1]:
                        swap_v = v_ptr[q * kk + j]
                        v_ptr[q * kk + j] = v_ptr[q * kk + j - 1]
                        v_ptr[q * kk + j - 1] = swap_v
                        swap_i = i_ptr[q * kk + j]
                        i_ptr[q * kk + j] = i_ptr[q * kk + j - 1]
                        i_ptr[q * kk + j - 1] = swap_i
                    else:
                        break
    else:
        for q in prange(Q, nogil=True, schedule='static'):
            for j in range(kk):
                i_ptr[q * kk + j] = j
                v_ptr[q * kk + j] = s_ptr[q * N + j]

            for j in range(kk - 1):
                m = j
                for n in range(j + 1, kk):
                    if v_ptr[q * kk + n] > v_ptr[q * kk + m]:
                        m = n
                if m != j:
                    swap_v = v_ptr[q * kk + j]
                    v_ptr[q * kk + j] = v_ptr[q * kk + m]
                    v_ptr[q * kk + m] = swap_v
                    swap_i = i_ptr[q * kk + j]
                    i_ptr[q * kk + j] = i_ptr[q * kk + m]
                    i_ptr[q * kk + m] = swap_i

            for n in range(kk, N):
                cand_v = s_ptr[q * N + n]
                if cand_v <= v_ptr[q * kk + kk - 1]:
                    continue
                v_ptr[q * kk + kk - 1] = cand_v
                i_ptr[q * kk + kk - 1] = n
                for j in range(kk - 1, 0, -1):
                    if v_ptr[q * kk + j] > v_ptr[q * kk + j - 1]:
                        swap_v = v_ptr[q * kk + j]
                        v_ptr[q * kk + j] = v_ptr[q * kk + j - 1]
                        v_ptr[q * kk + j - 1] = swap_v
                        swap_i = i_ptr[q * kk + j]
                        i_ptr[q * kk + j] = i_ptr[q * kk + j - 1]
                        i_ptr[q * kk + j - 1] = swap_i
                    else:
                        break

    return indices, values
