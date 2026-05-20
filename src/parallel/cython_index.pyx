# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange

ctypedef np.float32_t F32
ctypedef np.int32_t I32

cdef float NEG_INF_F32 = -3.402823466e38


cdef inline float _dot(
    const float* a,
    int oa,
    const float* b,
    int ob,
    int D
) noexcept nogil:
    cdef int d
    cdef float acc = 0.0
    for d in range(D):
        acc += a[oa + d] * b[ob + d]
    return acc


@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_top_k_omp(
    np.ndarray[F32, ndim=2, mode="c"] queries,
    np.ndarray[F32, ndim=2, mode="c"] index,
    int k,
    int num_threads=0,
):
    cdef int Q = queries.shape[0]
    cdef int N = index.shape[0]
    cdef int D = queries.shape[1]
    cdef int kk = k if k <= N else N
    cdef int threads = num_threads if num_threads > 0 else 1

    cdef int q, n, j, swap_i
    cdef float score, swap_v

    if index.shape[1] != D:
        raise ValueError("dimension mismatch")
    if kk <= 0:
        raise ValueError("k must be >= 1")

    cdef np.ndarray[I32, ndim=2, mode="c"] indices = np.empty((Q, kk), dtype=np.int32)
    cdef np.ndarray[F32, ndim=2, mode="c"] values = np.empty((Q, kk), dtype=np.float32)

    cdef const float* q_ptr = <const float*> queries.data
    cdef const float* i_ptr = <const float*> index.data
    cdef int* idx_ptr = <int*> indices.data
    cdef float* val_ptr = <float*> values.data

    with nogil:
        for q in prange(Q, schedule='static', num_threads=threads):
            for j in range(kk):
                idx_ptr[q * kk + j] = -1
                val_ptr[q * kk + j] = NEG_INF_F32

            for n in range(N):
                score = _dot(q_ptr, q * D, i_ptr, n * D, D)

                if score <= val_ptr[q * kk + kk - 1]:
                    continue

                val_ptr[q * kk + kk - 1] = score
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


@cython.boundscheck(False)
@cython.wraparound(False)
def top_k_descending(
    np.ndarray[F32, ndim=2, mode="c"] scores,
    int k,
    int num_threads=0,
):
    cdef int Q = scores.shape[0]
    cdef int N = scores.shape[1]
    cdef int kk = k if k <= N else N
    cdef int threads = num_threads if num_threads > 0 else 1

    cdef int q, j, n, m, swap_i
    cdef float cand_v, swap_v

    if kk <= 0:
        raise ValueError("k must be >= 1")

    cdef np.ndarray[I32, ndim=2, mode="c"] indices = np.empty((Q, kk), dtype=np.int32)
    cdef np.ndarray[F32, ndim=2, mode="c"] values = np.empty((Q, kk), dtype=np.float32)

    cdef const float* s_ptr = <const float*> scores.data
    cdef int* idx_ptr = <int*> indices.data
    cdef float* val_ptr = <float*> values.data

    with nogil:
        for q in prange(Q, schedule='static', num_threads=threads):
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


def cosine_top_k_auto(
    np.ndarray[F32, ndim=2, mode="c"] queries,
    np.ndarray[F32, ndim=2, mode="c"] index,
    int k,
    int num_threads=0,
    bint prefer_blas=True,
):
    if prefer_blas and queries.shape[0] <= 64:
        scores = np.ascontiguousarray(queries @ index.T, dtype=np.float32)
        return top_k_descending(scores, k, num_threads)
    return cosine_top_k_omp(queries, index, k, num_threads)