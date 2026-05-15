"""
Memory tokens for the CLaRiON compressor.

In CLaRa (Apple R&D), each document is concatenated with `l` learnable memory
tokens, and the final-layer hidden states at those positions form the
compressed document representation. We replicate this scheme with a small
trainable embedding table; in our compute-focused setting the table is
randomly initialized and held constant.

Reference: paper Section 2.2,
    M_i = LLM_theta_c([t_1,...,t_m, m_1,...,m_l])[m+1 : m+l]
"""

from __future__ import annotations

import numpy as np

from .config import ModelConfig


class MemoryTokens:
    """A small table of learnable memory-token embeddings.

    Stored as a (l, hidden) float32 array. The same tokens are reused for
    every document in the corpus, which is precisely what enables offline
    batched encoding.
    """

    __slots__ = ("config", "weights")

    def __init__(self, config: ModelConfig, weights: np.ndarray | None = None):
        self.config = config
        if weights is None:
            rng = np.random.default_rng(config.rng_seed + 1337)
            weights = rng.normal(
                0.0, config.init_scale,
                size=(config.n_memory_tokens, config.hidden_dim),
            ).astype(np.float32)
        assert weights.shape == (config.n_memory_tokens, config.hidden_dim)
        assert weights.dtype == np.float32
        self.weights = np.ascontiguousarray(weights)

    def expand_to_batch(self, batch_size: int) -> np.ndarray:
        """Broadcast the memory tokens across a batch.

        Returns a contiguous (B, l, hidden) array. We materialize it because
        downstream Cython kernels assume contiguous storage.
        """
        return np.broadcast_to(
            self.weights[None, :, :], (batch_size, *self.weights.shape)
        ).copy()
