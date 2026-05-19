"""
Memory tokens for the CLaRiON compressor.

In CLaRa (Apple R&D), each document is concatenated with `l` learnable memory
tokens, and the final-layer hidden states at those positions form the
compressed document representation.

We replicate this scheme with a fixed embedding table shared across all docs.

Reference:
    M_i = LLM_theta_c([t_1,...,t_m, m_1,...,m_l])[m+1 : m+l]
"""

from __future__ import annotations

import numpy as np

from .config import ModelConfig


class MemoryTokens:
    """
    Fixed memory-token embeddings shared across all documents.

    Shape:
        weights: (l, H) float32
    """

    __slots__ = ("config", "weights")

    def __init__(self, config: ModelConfig, weights: np.ndarray | None = None):
        self.config = config

        if weights is None:
            rng = np.random.default_rng(config.rng_seed + 1337)
            weights = rng.normal(
                0.0,
                config.init_scale,
                size=(config.n_memory_tokens, config.hidden_dim),
            ).astype(np.float32)

        weights = np.asarray(weights, dtype=np.float32)

        assert weights.shape == (
            config.n_memory_tokens,
            config.hidden_dim,
        ), f"Expected {(config.n_memory_tokens, config.hidden_dim)}, got {weights.shape}"

        self.weights = np.ascontiguousarray(weights)

    def expand_to_batch(self, batch_size: int) -> np.ndarray:
        """
        Expand memory tokens to batch dimension.

        Returns:
            (B, l, H) float32 contiguous array
        """

        if batch_size == 1:
            return self.weights[None, :, :].copy()

        return np.repeat(
            self.weights[None, :, :],
            repeats=batch_size,
            axis=0,
        ).astype(np.float32, copy=False)