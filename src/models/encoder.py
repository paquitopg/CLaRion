"""
CLaRiON Encoder (the "compressor" in CLaRa).

A tiny 2-layer transformer that compresses a document into a small set of
memory-token embeddings. Architecture mirrors Apple R&D's CLaRa
(https://github.com/apple/ml-clara), section 2.2, at toy scale:

    1.  Embed token ids of the document.
    2.  Append `n_memory_tokens` learnable memory tokens at the tail.
    3.  Run `n_layers` blocks of (causal self-attention + FFN), pre-norm.
    4.  Slice out the final-layer hidden states of the memory-token positions
        and return them flattened as a single (B, l*hidden) vector per doc.

Two execution backends are exposed:

- `EncoderNumpy`:        pure numpy. The "baseline" — implicitly BLAS-threaded
                         for the matmuls, but otherwise serial.
- `EncoderCython`:       same architecture, hot kernels delegated to the
                         OpenMP-parallel Cython module `src.parallel.cython_encoder`.

Per the project framing (Xavier's email: "vous n'avez pas besoin que le
modèle retourne des réponses correctes, juste qu'il aille plus vite"), the
weights are random-initialized once and the forward pass is exercised only
for timing.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import ModelConfig
from .memory import MemoryTokens

logger = logging.getLogger("clarion.encoder")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class TransformerLayerParams:
    """Parameters of a single transformer layer."""

    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray

    W1: np.ndarray
    W2: np.ndarray

    norm1: np.ndarray
    norm2: np.ndarray


@dataclass
class EncoderParams:
    """Full parameter set for the encoder."""

    embed: np.ndarray
    pos_embed: np.ndarray
    layers: list[TransformerLayerParams]
    norm_final: np.ndarray
    memory: MemoryTokens


def _init_params(config: ModelConfig) -> EncoderParams:
    """Initialize encoder parameters with Gaussian weights."""

    rng = np.random.default_rng(config.rng_seed)

    H = config.hidden_dim
    F = config.ffn_dim
    scale = config.init_scale

    max_len = config.max_seq_len + config.n_memory_tokens

    def randn(*shape: int) -> np.ndarray:
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    layers: list[TransformerLayerParams] = []

    for _ in range(config.n_layers):
        layers.append(
            TransformerLayerParams(
                Wq=randn(H, H),
                Wk=randn(H, H),
                Wv=randn(H, H),
                Wo=randn(H, H),
                W1=randn(H, F),
                W2=randn(F, H),
                norm1=np.ones(H, dtype=np.float32),
                norm2=np.ones(H, dtype=np.float32),
            )
        )

    return EncoderParams(
        embed=randn(config.vocab_size, H),
        pos_embed=randn(max_len, H),
        layers=layers,
        norm_final=np.ones(H, dtype=np.float32),
        memory=MemoryTokens(config),
    )


def _rms_norm(x: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    """RMS normalization."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _softmax_lastdim(x: np.ndarray) -> np.ndarray:
    """Stable softmax on last dimension."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attention_numpy(
    x: np.ndarray,
    layer: TransformerLayerParams,
    n_heads: int,
    head_dim: int,
    attention_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-head self-attention."""

    B, L, H = x.shape

    Q = x @ layer.Wq
    K = x @ layer.Wk
    V = x @ layer.Wv

    Q = Q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)

    if attention_mask is not None:
        scores = np.where(attention_mask[:, None, None, :], scores, -1e9)

    weights = _softmax_lastdim(scores)

    context = weights @ V

    out = context.transpose(0, 2, 1, 3).reshape(B, L, H)

    return out @ layer.Wo


def _ffn_numpy(x: np.ndarray, layer: TransformerLayerParams) -> np.ndarray:
    """Feed-forward network."""
    return _gelu(x @ layer.W1) @ layer.W2


def pool_memory(mem_states: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Pool memory token states into a single vector."""
    if mode == "mean":
        return mem_states.mean(axis=1)
    if mode == "first":
        return mem_states[:, 0]
    raise ValueError(mode)


class EncoderBackend:
    """Abstract encoder backend."""

    def __init__(
        self,
        config: ModelConfig,
        params: Optional[EncoderParams] = None,
    ):
        self.config = config
        self.params = params or _init_params(config)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Return memory token states."""
        raise NotImplementedError

    def encode_retrieval(
        self,
        token_ids: np.ndarray,
        pooling: str = "mean",
    ) -> np.ndarray:
        """Return pooled retrieval embedding."""
        return pool_memory(self.forward(token_ids), pooling)


class EncoderNumpy(EncoderBackend):
    """Pure NumPy encoder implementation."""

    def forward(self, token_ids: np.ndarray) -> np.ndarray:

        cfg = self.config
        B = token_ids.shape[0]

        doc_mask = token_ids != cfg.pad_id

        doc_h = self.params.embed[token_ids]
        mem = self.params.memory.expand_to_batch(B)

        x = np.concatenate([doc_h, mem], axis=1)

        pos = self.params.pos_embed[: x.shape[1]]
        x = x + pos[None, :, :]

        mem_mask = np.ones((B, cfg.n_memory_tokens), dtype=bool)
        attention_mask = np.concatenate([doc_mask, mem_mask], axis=1)

        for layer in self.params.layers:
            x = x + _attention_numpy(
                _rms_norm(x, layer.norm1, cfg.eps),
                layer,
                cfg.n_heads,
                cfg.head_dim,
                attention_mask,
            )

            x = x + _ffn_numpy(
                _rms_norm(x, layer.norm2, cfg.eps),
                layer,
            )

        x = _rms_norm(x, self.params.norm_final, cfg.eps)

        return np.ascontiguousarray(
            x[:, -cfg.n_memory_tokens :, :],
            dtype=np.float32,
        )


class EncoderCython(EncoderBackend):
    """Cython-accelerated encoder backend."""

    def __init__(
        self,
        config: ModelConfig,
        params: Optional[EncoderParams] = None,
        num_threads: int = 0,
    ):
        super().__init__(config, params)

        self.num_threads = num_threads
        self._available = False

        try:
            from src.parallel import cython_encoder
            self._ext = cython_encoder
            self._available = True
        except Exception as e:
            logger.warning("Cython unavailable: %s", e)
            self._ext = None

    def forward(self, token_ids: np.ndarray) -> np.ndarray:

        if not self._available:
            return EncoderNumpy(self.config, self.params).forward(token_ids)

        cfg = self.config
        B = token_ids.shape[0]

        doc_h = self.params.embed[token_ids]
        mem = self.params.memory.expand_to_batch(B)

        x = np.concatenate([doc_h, mem], axis=1)
        x = x + self.params.pos_embed[: x.shape[1]][None, :, :]
        x = np.ascontiguousarray(x, dtype=np.float32)

        for layer in self.params.layers:
            x = self._ext.encoder_block(
                x,
                layer.Wq,
                layer.Wk,
                layer.Wv,
                layer.Wo,
                layer.W1,
                layer.W2,
                layer.norm1,
                layer.norm2,
                cfg.n_heads,
                cfg.head_dim,
                cfg.eps,
                self.num_threads,
            )

        x = _rms_norm(x, self.params.norm_final, cfg.eps)

        return np.ascontiguousarray(
            x[:, -cfg.n_memory_tokens :, :],
            dtype=np.float32,
        )


def build_encoder(
    config: ModelConfig,
    backend: str = "numpy",
    num_threads: int = 0,
    params: Optional[EncoderParams] = None,
) -> EncoderBackend:
    """Factory for encoder backends."""

    if backend == "numpy":
        return EncoderNumpy(config, params)

    if backend == "cython":
        return EncoderCython(config, params, num_threads=num_threads)

    raise ValueError(f"Unknown backend: {backend}")