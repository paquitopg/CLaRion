"""
CLaRiON end-to-end glue.

Wires together the three modules that two different people own:

    Paco (encoder/index side)
        encoder.forward          : tokens -> doc embeddings (numpy)
        cosine_cython_omp        : query embeddings + bank -> scores (numpy)
        Retriever.search         : convenience wrapper over the above

    Avner (decoder side)
        ClaraTopK / ClaraTopKCython : differentiable ST top-k aggregator (torch)
        ClaraDecoder (WIP)          : generator

The two sides communicate at exactly one boundary: the similarity-score
matrix `S` of shape `(Q, N)`. This module handles:

  * the numpy <-> torch conversion (Paco's outputs are float32 numpy,
    Avner's modules consume torch.Tensors),
  * the bank materialization needed by the ST aggregator's
    `M_(k) = Z @ M` step (paper equation 3.6),
  * a single-function entry point for inference and a separate one for
    training that preserves the gradient path through `ClaraTopK`.

Use the inference path for the benchmarks; use the training path once
Avner's decoder is ready.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .index.scorer import cosine_cython_omp, cosine_numpy, top_k_indices
from .models.config import ModelConfig
from .models.encoder import EncoderBackend, build_encoder

if TYPE_CHECKING:  # pragma: no cover — torch import is heavy, defer.
    import torch
    from .models.topk import ClaraTopK

logger = logging.getLogger("clarion.pipeline")


# --------------------------------------------------------------------------- #
# Inference path (no gradients needed)
# --------------------------------------------------------------------------- #
@dataclass
class InferenceResult:
    """What the encoder side ships to the decoder side at inference time."""
    indices: np.ndarray         # (Q, k) int32 — which docs were chosen
    scores:  np.ndarray         # (Q, k) float32 — cos(q, M_{indices})
    top_k_embeddings: np.ndarray  # (Q, k, embedding_dim) float32 — gathered bank rows
    full_scores: Optional[np.ndarray] = None  # (Q, N) — kept if asked


def search(
    bank: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    backend: str = "cython",
    num_threads: int = 0,
    return_full_scores: bool = False,
) -> InferenceResult:
    """
    Retrieve top-k documents for a batch of query embeddings.

    Args
    ----
    bank             : (N, D) float32 — frozen document embedding bank
                       produced by the encoder.
    query_embeddings : (Q, D) float32 — already-encoded query memory tokens
                       (the decoder side calls this after running the query
                       reasoner LoRA).
    k                : how many docs to return per query.
    backend          : "cython" (OpenMP-parallel) or "numpy".
    num_threads      : OpenMP threads for the cython backend (0 = default).
    return_full_scores : also return the full (Q, N) similarity matrix —
                       needed if the caller plans to feed it to ClaraTopK
                       for the ST estimator.

    Returns
    -------
    InferenceResult with:
      - indices          : (Q, k) int32, sorted by score descending
      - scores           : (Q, k) float32, descending
      - top_k_embeddings : (Q, k, D) float32 — `bank` rows gathered at indices
      - full_scores      : (Q, N) if requested, else None
    """
    if backend == "numpy":
        S = cosine_numpy(query_embeddings, bank)
    elif backend == "cython":
        S = cosine_cython_omp(query_embeddings, bank, num_threads)
    else:
        raise ValueError(f"Unknown backend {backend!r}")

    idx, vals = top_k_indices(
        S, k=k,
        backend="cython" if backend == "cython" else "numpy",
        num_threads=num_threads,
    )

    # Gather the top-k bank rows. (Q, k, D)
    gathered = bank[idx]  # numpy fancy indexing: broadcasts over (Q, k)
    return InferenceResult(
        indices=idx,
        scores=vals,
        top_k_embeddings=gathered.astype(np.float32, copy=False),
        full_scores=S if return_full_scores else None,
    )


# --------------------------------------------------------------------------- #
# Training path (gradient flows through ClaraTopK back into the query encoder)
# --------------------------------------------------------------------------- #
def search_differentiable(
    bank: np.ndarray,
    query_embeddings: "torch.Tensor",
    topk_module: "ClaraTopK",
    num_threads: int = 0,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Training-time retrieval.

    `query_embeddings` is a torch.Tensor with `requires_grad=True` so that the
    decoder's NTP loss can flow back through the ST estimator to the query
    reasoner. The bank is *frozen* — CLaRa keeps the document embeddings
    fixed during end-to-end training (paper §3, "the compressor remains frozen
    to allow offline document encoding").

    Concretely:
        1. Compute scores S = cos(query_embeddings, bank). This step uses
           torch ops (not the Cython kernel) because we need grad w.r.t.
           query_embeddings.
        2. Feed S to Avner's `ClaraTopK` -> (Z, indices),
           with Z of shape (Q, k, N) carrying the ST gradient.
        3. Aggregate the top-k memory tokens:  M_(k) = Z @ bank  (eq. 3.6).
           This is the input the decoder will condition on.

    Returns
    -------
    (M_k, indices, Z)
        M_k     : (Q, k, D) torch.Tensor on the same device as queries
        indices : (Q, k)    torch.Tensor int — for debugging / logging
        Z       : (Q, k, N) torch.Tensor — the differentiable selector
    """
    import torch  # local import: don't make torch a hard dependency of the encoder side

    if not isinstance(query_embeddings, torch.Tensor):
        raise TypeError("search_differentiable expects a torch.Tensor for queries; "
                        "use `search()` for the numpy inference path.")

    # Bank stays as a torch tensor on the same device. No grad — it's frozen.
    bank_t = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32)).to(
        query_embeddings.device
    )

    # Cosine similarity in torch. Equivalent to `cosine_cython_omp` but with
    # an autograd graph leading back to query_embeddings.
    q_norm = torch.nn.functional.normalize(query_embeddings, dim=-1, eps=1e-12)
    b_norm = torch.nn.functional.normalize(bank_t, dim=-1, eps=1e-12)
    scores = q_norm @ b_norm.T  # (Q, N)

    # Run Avner's differentiable top-k (this is where the ST gradient lives).
    Z, indices = topk_module(scores)  # Z: (Q, k, N), indices: (Q, k)

    # Aggregate top-k bank rows. M_k[b, j, :] = sum_n Z[b, j, n] * bank[n, :]
    # Done as a single batched matmul.
    M_k = Z @ bank_t  # (Q, k, D)

    return M_k, indices, Z


# --------------------------------------------------------------------------- #
# Convenience builder
# --------------------------------------------------------------------------- #
def build_pipeline(
    config: ModelConfig,
    backend: str = "cython",
    num_threads: int = 0,
) -> EncoderBackend:
    """One-call builder for the encoder used at both indexing and query time."""
    return build_encoder(config, backend=backend, num_threads=num_threads)
