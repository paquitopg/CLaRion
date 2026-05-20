# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False

import numpy as np
cimport numpy as np

from libc.math cimport expf, sqrtf, tanhf

ctypedef np.float32_t F32

cdef float SQRT_2_OVER_PI = 0.7978845608


cdef inline void _softmax(float* x, int n) noexcept nogil:
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
    int dim,
    float eps
) noexcept nogil:
    cdef int i, j, base
    cdef float ss, inv
    for i in range(rows):
        base = i * dim
        ss = 0.0
        for j in range(dim):
            ss += x[base + j] * x[base + j]
        inv = 1.0 / sqrtf(ss / dim + eps)
        for j in range(dim):
            out[base + j] = x[base + j] * inv * scale[j]


cdef inline void _gelu(float* x, int n) noexcept nogil:
    cdef int i
    cdef float v
    for i in range(n):
        v = x[i]
        x[i] = 0.5 * v * (1.0 + tanhf(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))


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


cdef void _cross_attention_simple(
    float[:, :, ::1] x,
    float[:, :, ::1] mem,
    float[:, ::1] Wq,
    float[:, ::1] Wk,
    float[:, ::1] Wv,
    float[:, ::1] Wo,
    float[:, :, ::1] out
):
    cdef int B = x.shape[0]
    cdef int T = x.shape[1]
    cdef int S = mem.shape[1]
    cdef int H = x.shape[2]

    cdef np.ndarray[F32, ndim=3] Q_np = np.empty((B, T, H), dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] K_np = np.empty((B, S, H), dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] V_np = np.empty((B, S, H), dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] C_np = np.empty((B, T, H), dtype=np.float32)
    cdef np.ndarray[F32, ndim=2] scores_np = np.empty((T, S), dtype=np.float32)

    cdef float[:, :, ::1] Q = Q_np
    cdef float[:, :, ::1] K = K_np
    cdef float[:, :, ::1] V = V_np
    cdef float[:, :, ::1] C = C_np
    cdef float[:, ::1] scores = scores_np

    cdef int b, i, j, h, s
    cdef float scale = 1.0 / sqrtf(<float>H)
    cdef float acc

    for b in range(B):
        _matmul(&x[b, 0, 0], &Wq[0, 0], &Q[b, 0, 0], T, H, H)
        _matmul(&mem[b, 0, 0], &Wk[0, 0], &K[b, 0, 0], S, H, H)
        _matmul(&mem[b, 0, 0], &Wv[0, 0], &V[b, 0, 0], S, H, H)

        for i in range(T):
            for s in range(S):
                acc = 0.0
                for h in range(H):
                    acc += Q[b, i, h] * K[b, s, h]
                scores[i, s] = acc * scale

            _softmax(&scores[i, 0], S)

            for h in range(H):
                acc = 0.0
                for s in range(S):
                    acc += scores[i, s] * V[b, s, h]
                C[b, i, h] = acc

        _matmul(&C[b, 0, 0], &Wo[0, 0], &out[b, 0, 0], T, H, H)


cpdef np.ndarray[F32, ndim=3] decoder_forward_cython(
    np.ndarray[F32, ndim=3] x_in,
    np.ndarray[F32, ndim=3] mem_in,
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
    cdef np.ndarray[F32, ndim=3] x_np = np.ascontiguousarray(x_in, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] mem_np = np.ascontiguousarray(mem_in, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] tmp_np = np.empty_like(x_np)
    cdef np.ndarray[F32, ndim=3] attn_np = np.empty_like(x_np)
    cdef np.ndarray[F32, ndim=3] ff1_np = np.empty((x_np.shape[0], x_np.shape[1], W1.shape[1]), dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] ff2_np = np.empty_like(x_np)

    cdef float[:, :, ::1] x = x_np
    cdef float[:, :, ::1] mem = mem_np
    cdef float[:, :, ::1] tmp = tmp_np
    cdef float[:, :, ::1] attn = attn_np
    cdef float[:, :, ::1] ff1 = ff1_np
    cdef float[:, :, ::1] ff2 = ff2_np

    cdef float[:, ::1] Wqv = np.ascontiguousarray(Wq, dtype=np.float32)
    cdef float[:, ::1] Wkv = np.ascontiguousarray(Wk, dtype=np.float32)
    cdef float[:, ::1] Wvv = np.ascontiguousarray(Wv, dtype=np.float32)
    cdef float[:, ::1] Wov = np.ascontiguousarray(Wo, dtype=np.float32)
    cdef float[:, ::1] W1v = np.ascontiguousarray(W1, dtype=np.float32)
    cdef float[:, ::1] W2v = np.ascontiguousarray(W2, dtype=np.float32)
    cdef float[::1] norm1v = np.ascontiguousarray(norm1, dtype=np.float32)
    cdef float[::1] norm2v = np.ascontiguousarray(norm2, dtype=np.float32)

    cdef int B = x.shape[0]
    cdef int T = x.shape[1]
    cdef int H = x.shape[2]
    cdef int F = W1.shape[1]
    cdef int i, j, b

    _rms_norm(&x[0, 0, 0], &norm1v[0], &tmp[0, 0, 0], B * T, H, eps)
    _cross_attention_simple(tmp, mem, Wqv, Wkv, Wvv, Wov, attn)

    for b in range(B):
        for i in range(T):
            for j in range(H):
                x[b, i, j] += attn[b, i, j]

    _rms_norm(&x[0, 0, 0], &norm2v[0], &tmp[0, 0, 0], B * T, H, eps)

    _matmul(&tmp[0, 0, 0], &W1v[0, 0], &ff1[0, 0, 0], B * T, H, F)
    _gelu(&ff1[0, 0, 0], B * T * F)
    _matmul(&ff1[0, 0, 0], &W2v[0, 0], &ff2[0, 0, 0], B * T, F, H)

    for b in range(B):
        for i in range(T):
            for j in range(H):
                x[b, i, j] += ff2[b, i, j]

    return x_np


cpdef np.ndarray[F32, ndim=3] project_lm_head(
    np.ndarray[F32, ndim=3] hidden,
    np.ndarray[F32, ndim=2] lm_head
):
    cdef np.ndarray[F32, ndim=3] h = np.ascontiguousarray(hidden, dtype=np.float32)
    cdef np.ndarray[F32, ndim=2] w = np.ascontiguousarray(lm_head, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] logits = np.empty((h.shape[0], h.shape[1], w.shape[1]), dtype=np.float32)
    _matmul(&h[0, 0, 0], &w[0, 0], &logits[0, 0, 0], h.shape[0] * h.shape[1], h.shape[2], w.shape[1])
    return logits


cpdef void decoder_backward_lm_head(
    np.ndarray[F32, ndim=3] hidden,
    np.ndarray[F32, ndim=3] grad_logits,
    np.ndarray[F32, ndim=2] lm_head,
    float lr
):
    cdef int B = hidden.shape[0]
    cdef int T = hidden.shape[1]
    cdef int H = hidden.shape[2]
    cdef int V = grad_logits.shape[2]

    cdef np.ndarray[F32, ndim=2] h = np.ascontiguousarray(hidden, dtype=np.float32).reshape(B * T, H)
    cdef np.ndarray[F32, ndim=2] g = np.ascontiguousarray(grad_logits, dtype=np.float32).reshape(B * T, V)
    cdef np.ndarray[F32, ndim=2] grad_w = np.zeros((H, V), dtype=np.float32)

    cdef int i, j, k
    cdef float acc

    for i in range(H):
        for j in range(V):
            acc = 0.0
            for k in range(B * T):
                acc += h[k, i] * g[k, j]
            grad_w[i, j] = acc

    for i in range(H):
        for j in range(V):
            lm_head[i, j] -= lr * grad_w[i, j]