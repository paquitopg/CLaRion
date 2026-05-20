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

    vocab_size: int = 32_000
    pad_id: int = 0

    hidden_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 512

    n_memory_tokens: int = 8
    max_seq_len: int = 256

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
class TopKConfig:
    """
    Configuration for Top-K selection module.

    This module implements greedy Top-K selection used for sampling
    or structured sparsification of logits.
    """

    k: int = 8
    temperature: float = 1.0
    seed: int = 0

@dataclass(frozen=True)
class LossConfig:
    """
    Configuration for language modeling loss computation.

    Designed to be backend-agnostic (Python / NumPy / Cython).
    """

    ignore_index: int = 0
    eps: float = 1e-12
    use_cython: bool = True
    num_threads: int = 0


@dataclass(frozen=True)
class DecoderConfig:
    """
    Decoder configuration for a lightweight Transformer decoder.

    This configuration is fully independent from the encoder config.
    It defines all architectural and numerical hyperparameters required
    for NumPy and Cython backends.
    """

    hidden_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 512

    vocab_size: int = 32_000
    pad_id: int = 0

    eps: float = 1e-5
    init_scale: float = 0.02

    max_position_embeddings: int = 512
    default_temperature: float = 1.0
    default_topk: int = 0
    default_max_new_tokens: int = 50

    use_causal_mask: bool = True
    use_rms_norm: bool = True

    num_threads: int = 0

    @property
    def head_dim(self) -> int:
        """Dimension of a single attention head."""
        assert self.hidden_dim % self.n_heads == 0, \
            "hidden_dim must be divisible by n_heads"
        return self.hidden_dim // self.n_heads

    @property
    def scale_qk(self) -> float:
        """Attention scaling factor (optional explicit precompute)."""
        return self.head_dim ** -0.5

    @property
    def embedding_dim(self) -> int:
        """Embedding matrix dimension."""
        return self.hidden_dim

    @property
    def is_valid(self) -> bool:
        """Basic sanity check for configuration consistency."""
        return (
            self.hidden_dim > 0
            and self.n_heads > 0
            and self.ffn_dim > 0
            and self.vocab_size > 0
        )

@dataclass
class SystemConfig:
    encoder: ModelConfig
    decoder: DecoderConfig
    topk: TopKConfig
    backend: str = "numpy"