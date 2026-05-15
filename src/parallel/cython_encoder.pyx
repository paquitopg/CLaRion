# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: language_level=3
"""
CLaRiON Cython + OpenMP encoder kernels.

Hot kernels for the tiny-transformer compressor. Every public function is a
drop-in replacement for the corresponding numpy step in `EncoderNumpy`. The
parallelization strategy is:

  * Outer loop over the batch dimension (B) is parallelized with `prange`
    + `nogil`. Different documents are independent, so this is embarrassingly
    parallel and scales linearly with cores up to memory-bandwidth limits.
  * Within one document, the matmul / attention loops are plain C with -O3
    -march=native, which lets the compiler unroll + auto-vectorize (SIMD).

All buffers are float32 and assumed C-contiguous. Mutating intermediates are
allocated once outside the parallel region; per-thread scratch is taken via
slicing.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt, tanh
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free

ctypedef np.float32_t F32

cdef float SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi), for GELU


# --------------------------------------------------------------------------- #
# Low-level matmul: C = A @ B
# Shapes: A (M, K), B (K, N), C (M, N), all float32, C-contiguous.
# Parallelizes outer M loop; inner N/K loops are SIMD-friendly.
# --------------------------------------------------------------------------- #
cdef inline void _matmul_omp(
    float* A, float* B, float* C,
    int M, int K, int N,
    int num_threads,
) noexcept nogil:
    cdef int i, j, k
    cdef float acc
    cdef int nt = num_threads if num_threads > 0 else 0

    # Zero output. Cheap and lets us treat the inner loop as a pure accumulator.
    for i in range(M * N):
        C[i] = 0.0

    if nt > 0:
        for i in prange(M, nogil=True, schedule='static', num_threads=nt):
            for k in range(K):
                acc = A[i * K + k]
                for j in range(N):
                    C[i * N + j] += acc * B[k * N + j]
    else:
        for i in prange(M, nogil=True, schedule='static'):
            for k in range(K):
                acc = A[i * K + k]
                for j in range(N):
                    C[i * N + j] += acc * B[k * N + j]


# --------------------------------------------------------------------------- #
# RMSNorm over the last dimension, scaled by `scale` (H,).
# x: (B*L, H) flattened; modifies x in-place is *not* convenient because we
# need the un-normalized residual stream; instead we write into `out`.
# --------------------------------------------------------------------------- #
cdef inline void _rms_norm_rows(
    float* x, float* scale, float* out,
    int rows, int H, float eps, int num_threads,
) noexcept nogil:
    cdef int i, h
    cdef float ss, rms, inv
    cdef int nt = num_threads if num_threads > 0 else 0

    # NB: we write into `x[i*H+h] * x[i*H+h]` summed via `ss = ss + ...`
    # rather than `ss += ...`. Cython 3 treats `+=` on a scalar inside `prange`
    # as an OpenMP reduction, which then forbids reading the variable in the
    # same iteration body — but we do need to read it (to compute rms).
    if nt > 0:
        for i in prange(rows, nogil=True, schedule='static', num_threads=nt):
            ss = 0.0
            for h in range(H):
                ss = ss + x[i * H + h] * x[i * H + h]
            rms = sqrt(ss / H + eps)
            inv = 1.0 / rms
            for h in range(H):
                out[i * H + h] = x[i * H + h] * inv * scale[h]
    else:
        for i in prange(rows, nogil=True, schedule='static'):
            ss = 0.0
            for h in range(H):
                ss = ss + x[i * H + h] * x[i * H + h]
            rms = sqrt(ss / H + eps)
            inv = 1.0 / rms
            for h in range(H):
                out[i * H + h] = x[i * H + h] * inv * scale[h]


# --------------------------------------------------------------------------- #
# Approximate GELU applied in place to a row-major buffer of length n.
# --------------------------------------------------------------------------- #
cdef inline void _gelu_inplace(float* x, int n, int num_threads) noexcept nogil:
    cdef int i
    cdef float v
    cdef int nt = num_threads if num_threads > 0 else 0

    if nt > 0:
        for i in prange(n, nogil=True, schedule='static', num_threads=nt):
            v = x[i]
            x[i] = 0.5 * v * (1.0 + tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))
    else:
        for i in prange(n, nogil=True, schedule='static'):
            v = x[i]
            x[i] = 0.5 * v * (1.0 + tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))


# --------------------------------------------------------------------------- #
# Multi-head attention for ONE document (no batch dim).
# Inputs:
#   x          (L, H)   — input rows
#   Wq Wk Wv   (H, H)
#   Wo         (H, H)
# Output:
#   y          (L, H)   — attention output (already projected by Wo)
# Scratch:
#   tmp_qkv    >= 3 * L * H        (Q, K, V buffers)
#   tmp_scores >= n_heads * L * L  (per-head attention weights)
#   tmp_ctx    >= L * H            (context before Wo projection)
# Single-doc kernel: NOT thread-parallel internally (called from a parallel
# loop over the batch).
# --------------------------------------------------------------------------- #
cdef inline void _attention_single(
    float* x, float* Wq, float* Wk, float* Wv, float* Wo,
    int L, int H, int n_heads, int head_dim,
    float* tmp_qkv,    # 3 * L * H
    float* tmp_scores, # n_heads * L * L
    float* tmp_ctx,    # L * H
    float* y,          # L * H
) noexcept nogil:
    cdef int i, j, k, h_idx, t, row
    cdef float acc, scale, m, ss
    cdef float* Q = tmp_qkv
    cdef float* K = tmp_qkv + L * H
    cdef float* V = tmp_qkv + 2 * L * H

    # ---- Q = x @ Wq, K = x @ Wk, V = x @ Wv ----
    for row in range(L):
        for j in range(H):
            acc = 0.0
            for k in range(H):
                acc += x[row * H + k] * Wq[k * H + j]
            Q[row * H + j] = acc

            acc = 0.0
            for k in range(H):
                acc += x[row * H + k] * Wk[k * H + j]
            K[row * H + j] = acc

            acc = 0.0
            for k in range(H):
                acc += x[row * H + k] * Wv[k * H + j]
            V[row * H + j] = acc

    # ---- Per-head scaled dot-product attention ----
    scale = 1.0 / sqrt(<float>head_dim)
    for h_idx in range(n_heads):
        # scores[i, t] = Q[i, h_idx, :] dot K[t, h_idx, :] * scale
        for i in range(L):
            for t in range(L):
                acc = 0.0
                for k in range(head_dim):
                    acc += (
                        Q[i * H + h_idx * head_dim + k]
                        * K[t * H + h_idx * head_dim + k]
                    )
                tmp_scores[h_idx * L * L + i * L + t] = acc * scale

        # Softmax row-wise (numerically stable) over t.
        for i in range(L):
            m = tmp_scores[h_idx * L * L + i * L]
            for t in range(1, L):
                if tmp_scores[h_idx * L * L + i * L + t] > m:
                    m = tmp_scores[h_idx * L * L + i * L + t]
            ss = 0.0
            for t in range(L):
                tmp_scores[h_idx * L * L + i * L + t] = exp(
                    tmp_scores[h_idx * L * L + i * L + t] - m
                )
                ss += tmp_scores[h_idx * L * L + i * L + t]
            for t in range(L):
                tmp_scores[h_idx * L * L + i * L + t] /= ss

        # context[i, h_idx, :] = sum_t weights[i, t] * V[t, h_idx, :]
        for i in range(L):
            for k in range(head_dim):
                acc = 0.0
                for t in range(L):
                    acc += (
                        tmp_scores[h_idx * L * L + i * L + t]
                        * V[t * H + h_idx * head_dim + k]
                    )
                tmp_ctx[i * H + h_idx * head_dim + k] = acc

    # ---- y = ctx @ Wo ----
    for row in range(L):
        for j in range(H):
            acc = 0.0
            for k in range(H):
                acc += tmp_ctx[row * H + k] * Wo[k * H + j]
            y[row * H + j] = acc


# --------------------------------------------------------------------------- #
# Public: full encoder block forward.
#
# Pre-norm transformer block:
#     x = x + Attn(RMSNorm(x, norm1))
#     x = x + FFN (RMSNorm(x, norm2))
# Parallelism: outer batch loop with `prange`. Each thread owns its own
# scratch slab so writes are race-free.
# --------------------------------------------------------------------------- #
def encoder_block(
    np.ndarray[F32, ndim=3, mode="c"] x,
    np.ndarray[F32, ndim=2, mode="c"] Wq,
    np.ndarray[F32, ndim=2, mode="c"] Wk,
    np.ndarray[F32, ndim=2, mode="c"] Wv,
    np.ndarray[F32, ndim=2, mode="c"] Wo,
    np.ndarray[F32, ndim=2, mode="c"] W1,
    np.ndarray[F32, ndim=2, mode="c"] W2,
    np.ndarray[F32, ndim=1, mode="c"] norm1,
    np.ndarray[F32, ndim=1, mode="c"] norm2,
    int n_heads,
    int head_dim,
    float eps,
    int num_threads = 0,
):
    """
    Args
    ----
    x       : (B, L, H) input.  Modified-by-copy.
    Wq..Wo  : (H, H) attention projections.
    W1      : (H, F) FFN up.
    W2      : (F, H) FFN down.
    norm1   : (H,)   pre-attention RMSNorm scale.
    norm2   : (H,)   pre-FFN RMSNorm scale.
    n_heads, head_dim, eps : architecture knobs.
    num_threads : OpenMP threads (0 = let OpenMP pick).

    Returns
    -------
    out     : (B, L, H) updated hidden states (residual already added).
    """
    cdef int B = x.shape[0]
    cdef int L = x.shape[1]
    cdef int H = x.shape[2]
    cdef int F_ = W1.shape[1]

    cdef np.ndarray[F32, ndim=3] out = np.empty_like(x)
    cdef np.ndarray[F32, ndim=3] norm_buf = np.empty_like(x)
    cdef np.ndarray[F32, ndim=3] attn_buf = np.empty_like(x)
    # FFN intermediate buffer is (B, L, F_).
    cdef np.ndarray[F32, ndim=3] ffn_buf = np.empty((B, L, F_), dtype=np.float32)
    cdef np.ndarray[F32, ndim=3] ffn_out = np.empty_like(x)

    cdef float* x_ptr        = <float*> x.data
    cdef float* out_ptr      = <float*> out.data
    cdef float* normbuf_ptr  = <float*> norm_buf.data
    cdef float* attnbuf_ptr  = <float*> attn_buf.data
    cdef float* ffnbuf_ptr   = <float*> ffn_buf.data
    cdef float* ffnout_ptr   = <float*> ffn_out.data

    cdef float* Wq_ptr = <float*> Wq.data
    cdef float* Wk_ptr = <float*> Wk.data
    cdef float* Wv_ptr = <float*> Wv.data
    cdef float* Wo_ptr = <float*> Wo.data
    cdef float* W1_ptr = <float*> W1.data
    cdef float* W2_ptr = <float*> W2.data
    cdef float* n1_ptr = <float*> norm1.data
    cdef float* n2_ptr = <float*> norm2.data

    cdef int LH = L * H
    cdef int LF = L * F_

    # Per-document scratch (allocated once per call, sliced per thread).
    # Layout per doc:
    #   tmp_qkv    : 3 * L * H
    #   tmp_scores : n_heads * L * L
    #   tmp_ctx    :     L * H
    cdef int per_doc_qkv    = 3 * L * H
    cdef int per_doc_scores = n_heads * L * L
    cdef int per_doc_ctx    = L * H
    cdef int per_doc_total  = per_doc_qkv + per_doc_scores + per_doc_ctx

    cdef np.ndarray[F32, ndim=1] scratch = np.empty(B * per_doc_total, dtype=np.float32)
    cdef float* scratch_ptr = <float*> scratch.data

    cdef int b, i, j, k
    cdef float acc

    # === Stage 1: residual stream + attention ===
    # 1a. RMSNorm(x, norm1) into norm_buf.
    _rms_norm_rows(x_ptr, n1_ptr, normbuf_ptr, B * L, H, eps, num_threads)

    # 1b. Attention per document, in parallel across the batch.
    cdef int nt = num_threads if num_threads > 0 else 0
    if nt > 0:
        for b in prange(B, nogil=True, schedule='static', num_threads=nt):
            _attention_single(
                &normbuf_ptr[b * LH],
                Wq_ptr, Wk_ptr, Wv_ptr, Wo_ptr,
                L, H, n_heads, head_dim,
                &scratch_ptr[b * per_doc_total],
                &scratch_ptr[b * per_doc_total + per_doc_qkv],
                &scratch_ptr[b * per_doc_total + per_doc_qkv + per_doc_scores],
                &attnbuf_ptr[b * LH],
            )
    else:
        for b in prange(B, nogil=True, schedule='static'):
            _attention_single(
                &normbuf_ptr[b * LH],
                Wq_ptr, Wk_ptr, Wv_ptr, Wo_ptr,
                L, H, n_heads, head_dim,
                &scratch_ptr[b * per_doc_total],
                &scratch_ptr[b * per_doc_total + per_doc_qkv],
                &scratch_ptr[b * per_doc_total + per_doc_qkv + per_doc_scores],
                &attnbuf_ptr[b * LH],
            )

    # 1c. Residual: out = x + attn_buf
    for i in range(B * LH):
        out_ptr[i] = x_ptr[i] + attnbuf_ptr[i]

    # === Stage 2: residual stream + FFN ===
    # 2a. RMSNorm(out, norm2) into norm_buf.
    _rms_norm_rows(out_ptr, n2_ptr, normbuf_ptr, B * L, H, eps, num_threads)

    # 2b. ffn_buf = norm_buf @ W1     -> (B*L, F_)
    _matmul_omp(normbuf_ptr, W1_ptr, ffnbuf_ptr, B * L, H, F_, num_threads)

    # 2c. GELU in place.
    _gelu_inplace(ffnbuf_ptr, B * L * F_, num_threads)

    # 2d. ffn_out = ffn_buf @ W2     -> (B*L, H)
    _matmul_omp(ffnbuf_ptr, W2_ptr, ffnout_ptr, B * L, F_, H, num_threads)

    # 2e. Residual: out += ffn_out
    for i in range(B * LH):
        out_ptr[i] = out_ptr[i] + ffnout_ptr[i]

    return out
