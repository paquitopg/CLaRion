# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport expf, logf

ctypedef np.float32_t F32
ctypedef np.int32_t I32


cpdef float clara_lm_loss_cython_fast(
    np.ndarray[F32, ndim=3, mode="c"] logits,
    np.ndarray[I32, ndim=2, mode="c"] labels,
    int pad_id=0,
    int num_threads=0
):
    cdef int B = logits.shape[0]
    cdef int T = logits.shape[1]
    cdef int V = logits.shape[2]

    cdef int b, t, v, y
    cdef float total = 0.0
    cdef int count = 0
    cdef float* row
    cdef float maxv, sumexp

    for b in range(B):
        for t in range(T - 1):
            y = labels[b, t + 1]
            if y == pad_id:
                continue

            row = &logits[b, t, 0]
            maxv = row[0]

            for v in range(1, V):
                if row[v] > maxv:
                    maxv = row[v]

            sumexp = 0.0
            for v in range(V):
                sumexp += expf(row[v] - maxv)

            total += -(row[y] - maxv - logf(sumexp + 1e-12))
            count += 1

    if count == 0:
        return 0.0

    return total / count


cpdef tuple clara_ce_with_grad_cython(
    np.ndarray[F32, ndim=3, mode="c"] logits,
    np.ndarray[I32, ndim=2, mode="c"] labels,
    int ignore_index=0
):
    cdef int B = logits.shape[0]
    cdef int Tfull = logits.shape[1]
    cdef int T = Tfull - 1
    cdef int V = logits.shape[2]

    cdef np.ndarray[F32, ndim=3] grad = np.zeros((B, T, V), dtype=np.float32)
    cdef float[:, :, ::1] gv = grad

    cdef int b, t, v, y
    cdef float* row
    cdef float maxv, sumexp, p
    cdef float loss = 0.0
    cdef int count = 0
    cdef float inv_count

    for b in range(B):
        for t in range(T):
            y = labels[b, t + 1]
            if y == ignore_index:
                continue

            row = &logits[b, t, 0]
            maxv = row[0]

            for v in range(1, V):
                if row[v] > maxv:
                    maxv = row[v]

            sumexp = 0.0
            for v in range(V):
                sumexp += expf(row[v] - maxv)

            loss += -(row[y] - maxv - logf(sumexp + 1e-12))
            count += 1

            for v in range(V):
                p = expf(row[v] - maxv) / (sumexp + 1e-12)
                gv[b, t, v] = p

            gv[b, t, y] -= 1.0

    if count == 0:
        return 0.0, grad

    inv_count = 1.0 / count

    for b in range(B):
        for t in range(T):
            for v in range(V):
                gv[b, t, v] *= inv_count

    return loss * inv_count, grad