# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand

ctypedef np.float32_t F32
ctypedef np.int32_t I32


cdef inline void swap_f(float* a, float* b):
    cdef float tmp = a[0]
    a[0] = b[0]
    b[0] = tmp


cdef inline void swap_i(int* a, int* b):
    cdef int tmp = a[0]
    a[0] = b[0]
    b[0] = tmp


cdef int partition(float* arr, int* idx, int left, int right, int pivot):

    cdef float pivot_val = arr[pivot]

    swap_f(&arr[pivot], &arr[right])
    swap_i(&idx[pivot], &idx[right])

    cdef int store = left
    cdef int i

    for i in range(left, right):
        if arr[i] > pivot_val:
            swap_f(&arr[i], &arr[store])
            swap_i(&idx[i], &idx[store])
            store += 1

    swap_f(&arr[store], &arr[right])
    swap_i(&idx[store], &idx[right])

    return store


cdef void quickselect(float* arr, int* idx, int left, int right, int k):
    cdef int pivot, pos

    while left < right:
        pivot = left + (rand() % (right - left + 1))
        pos = partition(arr, idx, left, right, pivot)

        if pos == k:
            return
        elif pos > k:
            right = pos - 1
        else:
            left = pos + 1


cpdef np.ndarray[I32, ndim=2] greedy_topk_fast(
    np.ndarray[F32, ndim=2, mode="c"] logits,
    int k
):

    cdef int B = logits.shape[0]
    cdef int N = logits.shape[1]

    cdef np.ndarray[I32, ndim=2] out = np.empty((B, k), dtype=np.int32)
    cdef np.ndarray[I32, ndim=1] idx_row = np.empty(N, dtype=np.int32)

    cdef float* arr_ptr
    cdef int* idx_ptr

    cdef float[:, :] logits_mv = logits
    cdef int[:, :] out_mv = out
    cdef int[:] idx_mv = idx_row

    cdef int i, j

    for j in range(N):
        idx_row[j] = j

    for i in range(B):

        arr_ptr = &logits_mv[i, 0]
        idx_ptr = &idx_mv[0]

        quickselect(arr_ptr, idx_ptr, 0, N - 1, k)

        for j in range(k):
            out_mv[i, j] = idx_ptr[j]

    return out