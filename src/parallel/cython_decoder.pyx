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
    cdef float ss, inv, v

    for i in range(rows):
        base = i * dim
        ss = 0.0

        for j in range(dim):
            v = x[base + j]
            ss += v * v

        inv = 1.0 / sqrtf(ss / dim + eps)

        for j in range(dim):
            out[base + j] = x[base + j] * inv * scale[j]


cdef inline void _gelu(float* x, int n) noexcept nogil:
    cdef int i
    cdef float v

    for i in range(n):
        v = x[i]
        x[i] = 0.5 * v * (1.0 + tanhf(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))


cdef void _add_3d_inplace(
    float[:, :, ::1] x,
    float[:, :, ::1] y
) noexcept nogil:
    cdef int B = x.shape[0]
    cdef int T = x.shape[1]
    cdef int H = x.shape[2]
    cdef int b, t, h

    for b in range(B):
        for t in range(T):
            for h in range(H):
                x[b, t, h] += y[b, t, h]


cdef void _cross_attention_scores_context(
    float[:, :, ::1] Q,
    float[:, :, ::1] K,
    float[:, :, ::1] V,
    float[:, :, ::1] out
):
    cdef int B = Q.shape[0]
    cdef int T = Q.shape[1]
    cdef int S = K.shape[1]
    cdef int H = Q.shape[2]

    cdef np.ndarray[F32, ndim=2] scores_np = np.empty((T, S), dtype=np.float32)
    cdef float[:, ::1] scores = scores_np

    cdef int b, i, s, h
    cdef float acc
    cdef float scale = 1.0 / sqrtf(<float>H)

    for b in range(B):
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
                out[b, i, h] = acc


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
    cdef np.ndarray[F32, ndim=3] attn_ctx_np
    cdef np.ndarray[F32, ndim=3] attn_proj_np
    cdef np.ndarray[F32, ndim=2] ff1_2d
    cdef np.ndarray[F32, ndim=3] ff2_np

    cdef float[:, :, ::1] x = x_np
    cdef float[:, :, ::1] tmp = tmp_np
    cdef float[::1] norm1v = norm1
    cdef float[::1] norm2v = norm2

    cdef int B = x_np.shape[0]
    cdef int T = x_np.shape[1]
    cdef int H = x_np.shape[2]
    cdef int F = W1.shape[1]

    _rms_norm(&x[0, 0, 0], &norm1v[0], &tmp[0, 0, 0], B * T, H, eps)

    cdef np.ndarray[F32, ndim=3] Q_np = np.ascontiguousarray(tmp_np @ Wq, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] K_np = np.ascontiguousarray(mem_np @ Wk, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] V_np = np.ascontiguousarray(mem_np @ Wv, dtype=np.float32)

    attn_ctx_np = np.empty_like(x_np)
    _cross_attention_scores_context(Q_np, K_np, V_np, attn_ctx_np)

    attn_proj_np = np.ascontiguousarray(attn_ctx_np @ Wo, dtype=np.float32)
    _add_3d_inplace(x, attn_proj_np)

    _rms_norm(&x[0, 0, 0], &norm2v[0], &tmp[0, 0, 0], B * T, H, eps)

    ff1_2d = np.ascontiguousarray(tmp_np.reshape(B * T, H) @ W1, dtype=np.float32)
    _gelu(&ff1_2d[0, 0], B * T * F)

    ff2_np = np.ascontiguousarray((ff1_2d @ W2).reshape(B, T, H), dtype=np.float32)
    _add_3d_inplace(x, ff2_np)

    return x_np


cpdef np.ndarray[F32, ndim=3] project_lm_head(
    np.ndarray[F32, ndim=3] hidden,
    np.ndarray[F32, ndim=2] lm_head
):
    return np.ascontiguousarray(hidden @ lm_head, dtype=np.float32)


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
    cdef np.ndarray[F32, ndim=2] grad_w = np.ascontiguousarray(h.T @ g, dtype=np.float32)

    lm_head -= lr * grad_w