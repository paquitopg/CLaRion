"""
CLaRiON model configuration.

Mirrors the structural choices of Apple R&D's CLaRa
(https://github.com/apple/ml-clara), but at a toy scale: a 2-layer transformer
with a small hidden dimension. Per the project framing, the deliverable is a
CPU-parallelization study, so the architecture is deliberately small. Weights
are randomly initialized; correctness is not the goal, speed is.

The encoder (compressor) appends `n_memory_tokens` learnable memory tokens to
each document, runs `n_layers` of self-attention + FFN, and emits the final-
layer hidden states of just those memory-token positions as the document
embedding. This matches Section 2.2 of the CLaRa paper:

    M_i = LLM_theta_c([t_1, ..., t_m, m_1, ..., m_l])[m+1 : m+l]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Tiny transformer config shared by the encoder (compressor)."""

    # Vocabulary
    vocab_size: int = 32_000
    pad_id: int = 0

    # Architecture
    hidden_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 512

    # CLaRa-style continuous memory tokens
    n_memory_tokens: int = 8
    max_seq_len: int = 256

    # Numerical hygiene
    eps: float = 1e-5
    init_scale: float = 0.02
    rng_seed: int = 0

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.n_heads == 0, "hidden_dim must be divisible by n_heads"
        return self.hidden_dim // self.n_heads

    @property
    def embedding_dim(self) -> int:
        """Flat dim of one document's embedding: n_memory_tokens * hidden_dim."""
        return self.n_memory_tokens * self.hidden_dim


@dataclass(frozen=True)
class IndexConfig:
    """Configuration for the offline document index."""

    n_docs: int = 10_000
    batch_size: int = 64
    seed: int = 0
    # On-disk storage
    index_path: str = "data/index.npy"
    meta_path: str = "data/index_meta.json"


@dataclass(frozen=True)
class BenchConfig:
    """Knobs the benchmark scripts sweep over."""

    corpus_sizes: tuple = (1_000, 10_000, 50_000)
    thread_counts: tuple = (1, 2, 4, 8)
    n_queries: int = 32
    warmup_iters: int = 3
    measure_iters: int = 10

@dataclass(frozen=True)
class DecoderConfig:
    """
    Configuration STRICTEMENT decoder.
    Ne dépend pas du encoder ModelConfig.
    """

    hidden_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 512

    vocab_size: int = 32_000
    pad_id: int = 0

    eps: float = 1e-5
    init_scale: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads