# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np

from libc.math cimport expf, sqrtf, tanhf

ctypedef np.float32_t F32

cdef float SQRT_2_OVER_PI = 0.7978845608


cdef inline float fast_tanh(float x) noexcept nogil:
    cdef float x2 = x * x
    return x * (27.0 + x2) / (27.0 + 9.0 * x2)


cdef inline void _softmax_row(float* x, int n) noexcept nogil:
    cdef int i
    cdef float m = x[0]
    cdef float s = 0.0

    for i in range(1, n):
        if x[i] > m:
            m = x[i]

    for i in range(n):
        x[i] = expf(x[i] - m)
        s += x[i]

    for i in range(n):
        x[i] /= (s + 1e-12)


cdef inline void _rms_norm(
    float* x,
    float* scale,
    float* out,
    int rows,
    int H,
    float eps
) noexcept nogil:
    cdef int i, h, base
    cdef float ss, inv

    for i in range(rows):
        base = i * H
        ss = 0.0
        for h in range(H):
            ss += x[base + h] * x[base + h]

        inv = 1.0 / sqrtf(ss / H + eps)

        for h in range(H):
            out[base + h] = x[base + h] * inv * scale[h]


cdef inline void _gelu(float* x, int n) noexcept nogil:
    cdef int i
    cdef float v, inner

    for i in range(n):
        v = x[i]
        inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)
        x[i] = 0.5 * v * (1.0 + fast_tanh(inner))


cdef inline void _matmul(
    float* A,
    float* B,
    float* C,
    int M,
    int K,
    int N
) noexcept nogil:
    cdef int i, j, k
    cdef float acc

    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i * K + k] * B[k * N + j]
            C[i * N + j] = acc


cdef inline void _attention(
    float* x,
    float* memory,
    float* Wq,
    float* Wk,
    float* Wv,
    float* Wo,
    float* out,
    float* tmp_q,
    float* tmp_k,
    float* tmp_v,
    float* scores,
    float* ctx,
    int B,
    int T,
    int S,
    int H,
    int head_dim
) noexcept nogil:
    cdef int b, i, s, k, h
    cdef float acc
    cdef float scale = 1.0 / sqrtf(<float>head_dim)

    cdef int BT = B * T
    cdef int BS = B * S

    _matmul(x, Wq, tmp_q, BT, H, H)
    _matmul(memory, Wk, tmp_k, BS, H, H)
    _matmul(memory, Wv, tmp_v, BS, H, H)

    for b in range(B):
        for i in range(T):
            for s in range(S):
                acc = 0.0
                for k in range(head_dim):
                    acc += tmp_q[b * T * H + i * H + k] * tmp_k[b * S * H + s * H + k]
                scores[b * T * S + i * S + s] = acc * scale

            _softmax_row(&scores[b * T * S + i * S], S)

            for h in range(H):
                acc = 0.0
                for s in range(S):
                    acc += scores[b * T * S + i * S + s] * tmp_v[b * S * H + s * H + h]
                ctx[b * T * H + i * H + h] = acc

    _matmul(ctx, Wo, out, BT, H, H)


cpdef np.ndarray[F32, ndim=3] encoder_block(
    np.ndarray[F32, ndim=3] x_in,
    np.ndarray[F32, ndim=3] memory,
    np.ndarray[F32, ndim=2] Wq,
    np.ndarray[F32, ndim=2] Wk,
    np.ndarray[F32, ndim=2] Wv,
    np.ndarray[F32, ndim=2] Wo,
    np.ndarray[F32, ndim=2] W1,
    np.ndarray[F32, ndim=2] W2,
    np.ndarray[F32, ndim=1] norm1,
    np.ndarray[F32, ndim=1] norm2,
    float eps
):
    cdef np.ndarray[F32, ndim=3] x = np.ascontiguousarray(x_in, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] mem = np.ascontiguousarray(memory, dtype=np.float32)

    cdef int B = x.shape[0]
    cdef int T = x.shape[1]
    cdef int S = mem.shape[1]
    cdef int H = x.shape[2]
    cdef int F = W1.shape[1]

    cdef np.ndarray[F32, ndim=3] h1 = np.empty_like(x)
    cdef np.ndarray[F32, ndim=3] attn = np.empty_like(x)
    cdef np.ndarray[F32, ndim=2] tmp_q = np.empty((B * T, H), dtype=np.float32)
    cdef np.ndarray[F32, ndim=2] tmp_k = np.empty((B * S, H), dtype=np.float32)
    cdef np.ndarray[F32, ndim=2] tmp_v = np.empty((B * S, H), dtype=np.float32)
    cdef np.ndarray[F32, ndim=2] scores = np.empty((B * T, S), dtype=np.float32)
    cdef np.ndarray[F32, ndim=2] ctx = np.empty((B * T, H), dtype=np.float32)

    cdef np.ndarray[F32, ndim=3] h2 = np.empty_like(x)
    cdef np.ndarray[F32, ndim=2] ff1 = np.empty((B * T, F), dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] ff2 = np.empty_like(x)

    cdef int b, t, h

    _rms_norm(&x[0, 0, 0], &norm1[0], &h1[0, 0, 0], B * T, H, eps)

    _attention(
        &h1[0, 0, 0],
        &mem[0, 0, 0],
        &Wq[0, 0],
        &Wk[0, 0],
        &Wv[0, 0],
        &Wo[0, 0],
        &attn[0, 0, 0],
        &tmp_q[0, 0],
        &tmp_k[0, 0],
        &tmp_v[0, 0],
        &scores[0, 0],
        &ctx[0, 0],
        B, T, S, H, H
    )

    for b in range(B):
        for t in range(T):
            for h in range(H):
                x[b, t, h] += attn[b, t, h]

    _rms_norm(&x[0, 0, 0], &norm2[0], &h2[0, 0, 0], B * T, H, eps)

    _matmul(&h2[0, 0, 0], &W1[0, 0], &ff1[0, 0], B * T, H, F)
    _gelu(&ff1[0, 0], B * T * F)
    _matmul(&ff1[0, 0], &W2[0, 0], &ff2[0, 0, 0], B * T, F, H)

    for b in range(B):
        for t in range(T):
            for h in range(H):
                x[b, t, h] += ff2[b, t, h]

    return x