# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: nonecheck=False

import numpy as np
cimport numpy as np

from libc.math cimport exp
from cython.parallel cimport prange

ctypedef np.float32_t F32
ctypedef np.int32_t I32


# ============================================================
# OPTIMIZED DIFFERENTIABLE TOP-K
# ============================================================

cpdef differentiable_topk_full(
    np.ndarray[F32, ndim=2] logits,
    int k,
    float temperature
):

    cdef int B = logits.shape[0]
    cdef int N = logits.shape[1]

    cdef np.ndarray[F32, ndim=3] soft = np.zeros((B, k, N), dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] hard = np.zeros((B, k, N), dtype=np.float32)
    cdef np.ndarray[I32, ndim=2] indices = np.zeros((B, k), dtype=np.int32)

    cdef int i, j, n, best_idx
    cdef float best_val, val, max_val, denom

    cdef float[:, ::1] logits_mem = logits  # memoryview (important)

    # ========================================================
    # OPENMP PARALLEL BATCH
    # ========================================================

    for i in prange(B, nogil=True, schedule='static'):

        # local copy (IMPORTANT: avoid shared writes)
        cdef float taken_local[4096]   # assumes N <= 4096 (tu peux paramétrer)

        for n in range(N):
            taken_local[n] = 0.0

        # -----------------------
        # HARD TOP-K
        # -----------------------

        for j in range(k):

            best_val = -1e30
            best_idx = 0

            for n in range(N):

                if logits_mem[i, n] > best_val:
                    best_val = logits_mem[i, n]
                    best_idx = n

            indices[i, j] = best_idx
            hard[i, j, best_idx] = 1.0

            logits_mem[i, best_idx] = -1e30

        # -----------------------
        # SOFT TOP-K (FUSED PASS OPTIMIZED)
        # -----------------------

        for j in range(k):

            max_val = -1e30
            denom = 0.0

            # single pass max + exp accumulation (cache friendly)
            for n in range(N):

                if taken_local[n] < 0.5:

                    val = logits_mem[i, n]

                    if val > max_val:
                        max_val = val

            for n in range(N):

                if taken_local[n] < 0.5:

                    val = exp(logits_mem[i, n] - max_val)
                    soft[i, j, n] = val
                    denom += val

            for n in range(N):

                if taken_local[n] < 0.5:

                    soft[i, j, n] /= denom

            taken_local[indices[i, j]] = 1.0

    return soft + (hard - soft), indices