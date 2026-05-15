"""
Integration smoke test: encoder + index + (numpy-mock of) decoder top-k.

This script validates the *contract* between Paco's side and Avner's side
without needing torch installed (so it can run in any CI that has numpy +
Cython). It replicates Avner's `ClaraTopK.forward` in pure numpy, runs it
on the score matrix produced by my retrieval kernels, and asserts:

  (1) the score matrix matches the numpy reference (sanity);
  (2) the top-k indices selected by my Cython top-k match what
      a sorted-argpartition would have picked;
  (3) the differentiable `Z @ bank` aggregation produces a `(Q, k, D)`
      tensor with non-degenerate values (no NaN/inf, correct shape).

If all three hold, Paco's pipeline plugs into Avner's `ClaraTopK` with
nothing more than a `torch.from_numpy(scores)` wrapper at the boundary.

Run:

    PYTHONPATH=. python -m src.benchmarks.integration_check
"""

from __future__ import annotations

import logging

import numpy as np

from src.index.scorer import (
    cosine_cython_omp,
    cosine_numpy,
    top_k_indices,
)
from src.models.config import ModelConfig
from src.models.encoder import build_encoder

logging.getLogger("clarion").setLevel(logging.WARNING)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def clara_topk_numpy_mock(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Faithful numpy reproduction of Avner's ClaraTopK.forward.

    Returns (Z, indices, hard) where:
        Z       : (Q, k, N) — Straight-Through estimator output
        indices : (Q, k)    — descending by score (matches torch.topk)
        hard    : (Q, k, N) — one-hots at the chosen indices

    The math here is *exactly* what `ClaraTopK.forward` does in torch — we
    just don't have an autograd graph because numpy. The point is to confirm
    that the shapes, indices, and aggregation all behave as expected when
    the inputs come from my retrieval kernel.
    """
    Q, N = scores.shape
    idx_part = np.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
    chosen = np.take_along_axis(scores, idx_part, axis=-1)
    order = np.argsort(-chosen, axis=-1)
    indices = np.take_along_axis(idx_part, order, axis=-1).astype(np.int32)

    hard = np.zeros((Q, k, N), dtype=np.float32)
    for b in range(Q):
        for j in range(k):
            hard[b, j, indices[b, j]] = 1.0

    # Sequential masked softmax (Avner's iterative loop).
    soft = np.zeros_like(hard)
    taken = np.zeros((Q, N), dtype=np.float32)
    for j in range(k):
        mask = 1.0 - taken
        masked = scores + np.log(mask + 1e-8)
        soft[:, j] = _softmax(masked, axis=-1)
        taken = np.minimum(taken + hard[:, j], 1.0)

    # Straight-through: forward=hard, backward path would use soft.
    Z = hard + (soft - soft)  # no autograd here — symbolic shape match
    return Z, indices, hard


def main() -> None:
    cfg = ModelConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        n_memory_tokens=4, max_doc_len=32, vocab_size=4_000,
    )
    rng = np.random.default_rng(0)
    N_DOCS = 200
    N_QUERIES = 4
    K = 5

    # 1) Encode a small corpus and a few queries with shared params.
    encoder_numpy = build_encoder(cfg, backend="numpy")
    encoder_cy    = build_encoder(cfg, backend="cython", num_threads=2,
                                  params=encoder_numpy.params)

    docs    = rng.integers(1, cfg.vocab_size, size=(N_DOCS, cfg.max_doc_len), dtype=np.int32)
    queries = rng.integers(1, cfg.vocab_size, size=(N_QUERIES, cfg.max_doc_len), dtype=np.int32)

    bank_np = encoder_numpy.forward(docs)        # (N, D)
    bank_cy = encoder_cy.forward(docs)
    q_np    = encoder_numpy.forward(queries)     # (Q, D)
    q_cy    = encoder_cy.forward(queries)

    assert bank_np.shape == (N_DOCS, cfg.embedding_dim)
    assert q_np.shape == (N_QUERIES, cfg.embedding_dim)
    max_bank_diff = float(np.abs(bank_np - bank_cy).max())
    max_q_diff = float(np.abs(q_np - q_cy).max())
    print(f"[1/4] encoder numpy vs cython:   max|Δbank|={max_bank_diff:.2e}  max|Δq|={max_q_diff:.2e}")
    assert max_bank_diff < 5e-5 and max_q_diff < 5e-5

    # 2) Score the bank with both backends.
    S_np = cosine_numpy(q_np, bank_np)
    S_cy = cosine_cython_omp(q_np, bank_np, num_threads=2)
    max_S_diff = float(np.abs(S_np - S_cy).max())
    print(f"[2/4] cosine scores numpy vs cython:    max|ΔS|={max_S_diff:.2e}")
    assert max_S_diff < 1e-5

    # 3) Top-k via my Cython kernel vs numpy reference vs Avner's mock.
    idx_np, val_np  = top_k_indices(S_np, k=K, backend="numpy")
    idx_cy, val_cy  = top_k_indices(S_cy, k=K, backend="cython", num_threads=2)
    Z_mock, idx_mock, _ = clara_topk_numpy_mock(S_np, k=K)
    idx_match_cy = np.array_equal(idx_np, idx_cy)
    idx_match_mock = np.array_equal(idx_np, idx_mock)
    print(f"[3/4] top-k indices: cython == numpy? {idx_match_cy}   "
          f"ClaraTopK-mock == numpy? {idx_match_mock}")
    assert idx_match_cy and idx_match_mock, (
        "Top-k inconsistency between backends — would break ST aggregation."
    )

    # 4) Aggregate top-k bank rows the way Avner's decoder will (Z @ bank).
    M_k = Z_mock @ bank_np  # (Q, K, D)
    print(f"[4/4] aggregated M_k shape={M_k.shape}, "
          f"has_nan={np.isnan(M_k).any()}, has_inf={np.isinf(M_k).any()}")
    assert M_k.shape == (N_QUERIES, K, cfg.embedding_dim)
    assert not np.isnan(M_k).any() and not np.isinf(M_k).any()

    print("\nIntegration OK — encoder/index outputs plug into ClaraTopK contract.")


if __name__ == "__main__":
    main()
