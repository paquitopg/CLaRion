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
from libc.math cimport expf, sqrtf, tanhf

ctypedef np.float32_t F32
ctypedef np.uint8_t U8

cdef float SQRT_2_OVER_PI = 0.7978845608
cdef float NEG_INF = -1e30


cdef inline float fast_gelu_scalar(float x) noexcept nogil:
    return 0.5 * x * (1.0 + tanhf(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


cdef inline void _rms_norm_3d_parallel(
    float[:, :, ::1] x,
    float[::1] scale,
    float[:, :, ::1] out,
    float eps,
    int num_threads
) noexcept nogil:
    cdef int B = x.shape[0]
    cdef int L = x.shape[1]
    cdef int H = x.shape[2]
    cdef int b, t, h
    cdef float ss, inv, tmp

    for b in prange(B, nogil=True, schedule='static', num_threads=num_threads):
        for t in range(L):
            ss = 0.0
            for h in range(H):
                tmp = x[b, t, h]
                ss = ss + tmp * tmp
            inv = 1.0 / sqrtf(ss / H + eps)
            for h in range(H):
                out[b, t, h] = x[b, t, h] * inv * scale[h]


cdef inline void _add_inplace_3d_parallel(
    float[:, :, ::1] x,
    float[:, :, ::1] y,
    int num_threads
) noexcept nogil:
    cdef int B = x.shape[0]
    cdef int L = x.shape[1]
    cdef int H = x.shape[2]
    cdef int b, t, h

    for b in prange(B, nogil=True, schedule='static', num_threads=num_threads):
        for t in range(L):
            for h in range(H):
                x[b, t, h] = x[b, t, h] + y[b, t, h]


cdef inline void _gelu_inplace_2d_parallel(
    float[:, ::1] x,
    int num_threads
) noexcept nogil:
    cdef int M = x.shape[0]
    cdef int N = x.shape[1]
    cdef int i, j

    for i in prange(M, nogil=True, schedule='static', num_threads=num_threads):
        for j in range(N):
            x[i, j] = fast_gelu_scalar(x[i, j])


cdef void _streaming_attention_blockwise_parallel(
    float[:, :, ::1] Q,
    float[:, :, ::1] K,
    float[:, :, ::1] V,
    U8[:, ::1] mask,
    float[:, :, ::1] ctx,
    int n_heads,
    int head_dim,
    int block_size,
    int num_threads
) noexcept nogil:
    cdef int B = Q.shape[0]
    cdef int L = Q.shape[1]

    cdef int bt, b, t
    cdef int hh, s, d, s0, s1, off
    cdef float scale = 1.0 / sqrtf(<float>head_dim)
    cdef float score
    cdef float m_old, m_new, d_old, d_new, w, alpha
    cdef float acc

    for bt in prange(B * L, nogil=True, schedule='static', num_threads=num_threads):
        b = bt // L
        t = bt % L

        for hh in range(n_heads):
            off = hh * head_dim
            m_old = NEG_INF
            d_old = 0.0

            for d in range(head_dim):
                ctx[b, t, off + d] = 0.0

            s0 = 0
            while s0 < L:
                s1 = s0 + block_size
                if s1 > L:
                    s1 = L

                m_new = m_old

                for s in range(s0, s1):
                    if mask[b, s] == 0:
                        continue

                    acc = 0.0
                    for d in range(head_dim):
                        acc = acc + Q[b, t, off + d] * K[b, s, off + d]
                    score = acc * scale

                    if score > m_new:
                        m_new = score

                if m_new == NEG_INF:
                    s0 = s1
                    continue

                if m_old == NEG_INF:
                    alpha = 0.0
                else:
                    alpha = expf(m_old - m_new)

                for d in range(head_dim):
                    ctx[b, t, off + d] = ctx[b, t, off + d] * alpha

                d_new = d_old * alpha

                for s in range(s0, s1):
                    if mask[b, s] == 0:
                        continue

                    acc = 0.0
                    for d in range(head_dim):
                        acc = acc + Q[b, t, off + d] * K[b, s, off + d]
                    score = acc * scale

                    w = expf(score - m_new)
                    d_new = d_new + w

                    for d in range(head_dim):
                        ctx[b, t, off + d] = ctx[b, t, off + d] + w * V[b, s, off + d]

                m_old = m_new
                d_old = d_new
                s0 = s1

            if d_old > 0.0:
                for d in range(head_dim):
                    ctx[b, t, off + d] = ctx[b, t, off + d] / d_old


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[F32, ndim=3] encoder_block_hybrid_blockwise(
    np.ndarray[F32, ndim=3, mode="c"] x_in,
    np.ndarray[F32, ndim=2, mode="c"] Wq,
    np.ndarray[F32, ndim=2, mode="c"] Wk,
    np.ndarray[F32, ndim=2, mode="c"] Wv,
    np.ndarray[F32, ndim=2, mode="c"] Wo,
    np.ndarray[F32, ndim=2, mode="c"] W1,
    np.ndarray[F32, ndim=2, mode="c"] W2,
    np.ndarray[F32, ndim=1, mode="c"] norm1,
    np.ndarray[F32, ndim=1, mode="c"] norm2,
    np.ndarray[U8, ndim=2, mode="c"] attention_mask,
    int n_heads,
    int head_dim,
    float eps,
    int num_threads=1,
    int block_size=64,
):
    cdef int B = x_in.shape[0]
    cdef int L = x_in.shape[1]
    cdef int H = x_in.shape[2]

    if H != n_heads * head_dim:
        raise ValueError("hidden_dim must equal n_heads * head_dim")
    if num_threads <= 0:
        num_threads = 1
    if block_size <= 0:
        block_size = 64

    cdef np.ndarray[F32, ndim=3] x = np.ascontiguousarray(x_in, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] h1 = np.empty_like(x)
    cdef np.ndarray[F32, ndim=3] h2 = np.empty_like(x)
    cdef np.ndarray[F32, ndim=3] ctx = np.empty_like(x)

    cdef float[:, :, ::1] xv = x
    cdef float[:, :, ::1] h1v = h1
    cdef float[:, :, ::1] h2v = h2
    cdef float[:, :, ::1] ctxv = ctx
    cdef float[::1] n1 = norm1
    cdef float[::1] n2 = norm2
    cdef U8[:, ::1] mask = attention_mask

    with nogil:
        _rms_norm_3d_parallel(xv, n1, h1v, eps, num_threads)

    cdef np.ndarray[F32, ndim=3] Q = np.ascontiguousarray(h1 @ Wq, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] K = np.ascontiguousarray(h1 @ Wk, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] V = np.ascontiguousarray(h1 @ Wv, dtype=np.float32)

    cdef float[:, :, ::1] Qv = Q
    cdef float[:, :, ::1] Kv = K
    cdef float[:, :, ::1] Vv = V

    with nogil:
        _streaming_attention_blockwise_parallel(
            Qv, Kv, Vv, mask, ctxv,
            n_heads, head_dim, block_size, num_threads
        )

    cdef np.ndarray[F32, ndim=3] proj = np.ascontiguousarray(ctx @ Wo, dtype=np.float32)
    cdef float[:, :, ::1] projv = proj

    with nogil:
        _add_inplace_3d_parallel(xv, projv, num_threads)
        _rms_norm_3d_parallel(xv, n2, h2v, eps, num_threads)

    cdef np.ndarray[F32, ndim=2] ff1 = np.ascontiguousarray(
        h2.reshape(B * L, H) @ W1,
        dtype=np.float32
    )
    cdef float[:, ::1] ff1v = ff1

    with nogil:
        _gelu_inplace_2d_parallel(ff1v, num_threads)

    cdef np.ndarray[F32, ndim=2] ff2_2d = np.ascontiguousarray(ff1 @ W2, dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] ff2 = np.ascontiguousarray(ff2_2d.reshape(B, L, H), dtype=np.float32)
    cdef float[:, :, ::1] ff2v = ff2

    with nogil:
        _add_inplace_3d_parallel(xv, ff2v, num_threads)

    return x