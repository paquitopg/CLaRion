# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt
from cython.parallel cimport prange

ctypedef np.float32_t F32
ctypedef np.int32_t I32


# =========================================================
# COSINE SCORES
# =========================================================

def cosine_scores_omp(
    np.ndarray[F32, ndim=2, mode="c"] queries,
    np.ndarray[F32, ndim=2, mode="c"] index,
    int num_threads=0,
):
    """
    queries : (Q, D)
    index   : (N, D)

    returns : (Q, N)
    """

    cdef int Q = queries.shape[0]
    cdef int N = index.shape[0]
    cdef int D = queries.shape[1]

    if index.shape[1] != D:
        raise ValueError("dimension mismatch")

    cdef np.ndarray[F32, ndim=2] scores = np.empty(
        (Q, N),
        dtype=np.float32,
    )

    cdef np.ndarray[F32, ndim=1] qnorms = np.empty(
        Q,
        dtype=np.float32,
    )

    cdef np.ndarray[F32, ndim=1] inorms = np.empty(
        N,
        dtype=np.float32,
    )

    cdef float* q_ptr = <float*>queries.data
    cdef float* i_ptr = <float*>index.data
    cdef float* s_ptr = <float*>scores.data

    cdef float* qn_ptr = <float*>qnorms.data
    cdef float* in_ptr = <float*>inorms.data

    cdef int q, n, d
    cdef float acc
    cdef float qnorm

    # -----------------------------------------------------
    # QUERY NORMS
    # -----------------------------------------------------

    if num_threads > 0:

        for q in prange(
            Q,
            nogil=True,
            schedule='static',
            num_threads=num_threads,
        ):

            acc = 0.0

            for d in range(D):
                acc = acc + (
                    q_ptr[q * D + d] *
                    q_ptr[q * D + d]
                )

            qn_ptr[q] = sqrt(acc) + 1e-12

    else:

        for q in prange(
            Q,
            nogil=True,
            schedule='static',
        ):

            acc = 0.0

            for d in range(D):
                acc = acc + (
                    q_ptr[q * D + d] *
                    q_ptr[q * D + d]
                )

            qn_ptr[q] = sqrt(acc) + 1e-12

    # -----------------------------------------------------
    # INDEX NORMS
    # -----------------------------------------------------

    if num_threads > 0:

        for n in prange(
            N,
            nogil=True,
            schedule='static',
            num_threads=num_threads,
        ):

            acc = 0.0

            for d in range(D):
                acc = acc + (
                    i_ptr[n * D + d] *
                    i_ptr[n * D + d]
                )

            in_ptr[n] = sqrt(acc) + 1e-12

    else:

        for n in prange(
            N,
            nogil=True,
            schedule='static',
        ):

            acc = 0.0

            for d in range(D):
                acc = acc + (
                    i_ptr[n * D + d] *
                    i_ptr[n * D + d]
                )

            in_ptr[n] = sqrt(acc) + 1e-12

    # -----------------------------------------------------
    # MAIN COSINE
    # -----------------------------------------------------

    for q in range(Q):

        qnorm = qn_ptr[q]

        if num_threads > 0:

            for n in prange(
                N,
                nogil=True,
                schedule='static',
                num_threads=num_threads,
            ):

                acc = 0.0

                for d in range(D):
                    acc = acc + (
                        q_ptr[q * D + d] *
                        i_ptr[n * D + d]
                    )

                s_ptr[q * N + n] = (
                    acc / (qnorm * in_ptr[n])
                )

        else:

            for n in prange(
                N,
                nogil=True,
                schedule='static',
            ):

                acc = 0.0

                for d in range(D):
                    acc = acc + (
                        q_ptr[q * D + d] *
                        i_ptr[n * D + d]
                    )

                s_ptr[q * N + n] = (
                    acc / (qnorm * in_ptr[n])
                )

    return scores


# =========================================================
# TOP-K
# =========================================================

def top_k_descending(
    np.ndarray[F32, ndim=2, mode="c"] scores,
    int k,
    int num_threads=0,
):
    """
    scores : (Q, N)

    returns:
        indices : (Q, k)
        values  : (Q, k)
    """

    cdef int Q = scores.shape[0]
    cdef int N = scores.shape[1]

    cdef int kk = k if k <= N else N

    cdef np.ndarray[I32, ndim=2] indices = np.empty(
        (Q, kk),
        dtype=np.int32,
    )

    cdef np.ndarray[F32, ndim=2] values = np.empty(
        (Q, kk),
        dtype=np.float32,
    )

    cdef float* s_ptr = <float*>scores.data
    cdef int* idx_ptr = <int*>indices.data
    cdef float* val_ptr = <float*>values.data

    cdef int q, j, n, m
    cdef int swap_i

    cdef float cand_v
    cdef float swap_v

    if num_threads > 0:

        for q in prange(
            Q,
            nogil=True,
            schedule='static',
            num_threads=num_threads,
        ):

            # init
            for j in range(kk):
                idx_ptr[q * kk + j] = j
                val_ptr[q * kk + j] = s_ptr[q * N + j]

            # initial sort
            for j in range(kk - 1):

                m = j

                for n in range(j + 1, kk):

                    if val_ptr[q * kk + n] > val_ptr[q * kk + m]:
                        m = n

                if m != j:

                    swap_v = val_ptr[q * kk + j]
                    val_ptr[q * kk + j] = val_ptr[q * kk + m]
                    val_ptr[q * kk + m] = swap_v

                    swap_i = idx_ptr[q * kk + j]
                    idx_ptr[q * kk + j] = idx_ptr[q * kk + m]
                    idx_ptr[q * kk + m] = swap_i

            # streaming update
            for n in range(kk, N):

                cand_v = s_ptr[q * N + n]

                if cand_v <= val_ptr[q * kk + kk - 1]:
                    continue

                val_ptr[q * kk + kk - 1] = cand_v
                idx_ptr[q * kk + kk - 1] = n

                for j in range(kk - 1, 0, -1):

                    if val_ptr[q * kk + j] > val_ptr[q * kk + j - 1]:

                        swap_v = val_ptr[q * kk + j]
                        val_ptr[q * kk + j] = val_ptr[q * kk + j - 1]
                        val_ptr[q * kk + j - 1] = swap_v

                        swap_i = idx_ptr[q * kk + j]
                        idx_ptr[q * kk + j] = idx_ptr[q * kk + j - 1]
                        idx_ptr[q * kk + j - 1] = swap_i

                    else:
                        break

    else:

        for q in prange(
            Q,
            nogil=True,
            schedule='static',
        ):

            for j in range(kk):
                idx_ptr[q * kk + j] = j
                val_ptr[q * kk + j] = s_ptr[q * N + j]

            for j in range(kk - 1):

                m = j

                for n in range(j + 1, kk):

                    if val_ptr[q * kk + n] > val_ptr[q * kk + m]:
                        m = n

                if m != j:

                    swap_v = val_ptr[q * kk + j]
                    val_ptr[q * kk + j] = val_ptr[q * kk + m]
                    val_ptr[q * kk + m] = swap_v

                    swap_i = idx_ptr[q * kk + j]
                    idx_ptr[q * kk + j] = idx_ptr[q * kk + m]
                    idx_ptr[q * kk + m] = swap_i

            for n in range(kk, N):

                cand_v = s_ptr[q * N + n]

                if cand_v <= val_ptr[q * kk + kk - 1]:
                    continue

                val_ptr[q * kk + kk - 1] = cand_v
                idx_ptr[q * kk + kk - 1] = n

                for j in range(kk - 1, 0, -1):

                    if val_ptr[q * kk + j] > val_ptr[q * kk + j - 1]:

                        swap_v = val_ptr[q * kk + j]
                        val_ptr[q * kk + j] = val_ptr[q * kk + j - 1]
                        val_ptr[q * kk + j - 1] = swap_v

                        swap_i = idx_ptr[q * kk + j]
                        idx_ptr[q * kk + j] = idx_ptr[q * kk + j - 1]
                        idx_ptr[q * kk + j - 1] = swap_i

                    else:
                        break

    return indices, values