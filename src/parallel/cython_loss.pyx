# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport exp, log

ctypedef np.float32_t F32
ctypedef np.int32_t I32


cdef inline float _row_loss(
    float* row,
    int target,
    int V
) nogil:

    cdef int v
    cdef float maxv = row[0]
    cdef float acc = 0.0
    cdef float inv
    cdef float val

    for v in range(1, V):
        if row[v] > maxv:
            maxv = row[v]


    val = row[target]

    for v in range(V):
        acc += exp(row[v] - maxv)


    return -(val - log(acc + 1e-12))


cpdef float clara_lm_loss_cython_fast(
    np.ndarray[F32, ndim=3, mode="c"] logits,
    np.ndarray[I32, ndim=2, mode="c"] labels,
    int pad_id = 0,
    int num_threads = 0
):

    cdef int B = logits.shape[0]
    cdef int T = logits.shape[1]
    cdef int V = logits.shape[2]

    cdef float total = 0.0
    cdef int count = 0

    cdef int b, t
    cdef float* base
    cdef float loss

    cdef int nt = num_threads if num_threads > 0 else 0
    cdef int idx


    for b in prange(B, nogil=True, schedule='static', num_threads=nt):

        base = &logits[b, 0, 0]

        for t in range(T - 1):

            if labels[b, t + 1] == pad_id:
                continue

            idx = t * V
            loss = _row_loss(&base[idx], labels[b, t + 1], V)

            total += loss
            count += 1

    if count == 0:
        return 0.0

    return total / count