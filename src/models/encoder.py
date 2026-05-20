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
    handler.setStream(handler.stream)
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

        H = self.config.hidden_dim
        self.retrieval_proj = np.eye(H, dtype=np.float32)
        self.query_bias = np.zeros(H, dtype=np.float32)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Return memory token states."""
        raise NotImplementedError

    def encode_retrieval(
        self,
        token_ids: np.ndarray,
        pooling: str = "mean",
    ) -> np.ndarray:
        """Return pooled retrieval embedding (with trainable head)."""
        raise NotImplementedError

    def backward_query(
        self,
        token_ids: np.ndarray,
        grad_query: np.ndarray,
        lr: float = 1e-3,
    ) -> None:
        """
        Update retrieval head given gradient w.r.t. query.

        grad_query: dL/dquery, shape (B, H)
        Updated:
          - self.retrieval_proj (H, H)
          - self.query_bias (H,)
        """
        raise NotImplementedError


class EncoderNumpy(EncoderBackend):
    """Pure NumPy encoder implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_pooled: np.ndarray | None = None

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

    def encode_retrieval(
        self,
        token_ids: np.ndarray,
        pooling: str = "mean",
    ) -> np.ndarray:
        mem_states = self.forward(token_ids)
        pooled = pool_memory(mem_states, pooling)

        self._last_pooled = pooled

        query = pooled @ self.retrieval_proj + self.query_bias
        query = np.ascontiguousarray(query, dtype=np.float32)

        return query

    def backward_query(
        self,
        token_ids: np.ndarray,
        grad_query: np.ndarray,
        lr: float = 1e-3,
    ) -> None:
        """
        Update retrieval_proj and query_bias using grad_query.

        grad_query: dL/dquery, shape (B, H)
        """
        pooled = self._last_pooled
        if pooled is None:

            mem_states = self.forward(token_ids)
            pooled = pool_memory(mem_states, "mean")
            self._last_pooled = pooled

        B, H = pooled.shape

        grad_W = pooled.T @ grad_query / max(B, 1)
        b = grad_query.mean(axis=0)

        self.retrieval_proj -= lr * grad_W.astype(np.float32)
        self.query_bias -= lr * b.astype(np.float32)


class EncoderCython(EncoderBackend):
    def __init__(
        self,
        config: ModelConfig,
        params: Optional[EncoderParams] = None,
        num_threads: int = 0,
    ):
        super().__init__(config, params)
        self.num_threads = num_threads if num_threads and num_threads > 0 else 1
        self._available = False
        self._params_prepared = False
        self._last_pooled: np.ndarray | None = None

        try:
            from src.parallel import cython_encoder
            self._ext = cython_encoder
            self._available = True
        except Exception as e:
            logger.warning("Cython unavailable: %s", e)
            self._ext = None

        if self.params is not None:
            self._prepare_contiguous_params()

    def _prepare_contiguous_params(self) -> None:
        if self.params is None or self._params_prepared:
            return

        p = self.params

        p.embed = np.ascontiguousarray(p.embed, dtype=np.float32)
        p.pos_embed = np.ascontiguousarray(p.pos_embed, dtype=np.float32)
        p.norm_final = np.ascontiguousarray(p.norm_final, dtype=np.float32)

        for layer in p.layers:
            layer.Wq = np.ascontiguousarray(layer.Wq, dtype=np.float32)
            layer.Wk = np.ascontiguousarray(layer.Wk, dtype=np.float32)
            layer.Wv = np.ascontiguousarray(layer.Wv, dtype=np.float32)
            layer.Wo = np.ascontiguousarray(layer.Wo, dtype=np.float32)
            layer.W1 = np.ascontiguousarray(layer.W1, dtype=np.float32)
            layer.W2 = np.ascontiguousarray(layer.W2, dtype=np.float32)
            layer.norm1 = np.ascontiguousarray(layer.norm1, dtype=np.float32)
            layer.norm2 = np.ascontiguousarray(layer.norm2, dtype=np.float32)

        self._params_prepared = True

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if not self._available:
            return EncoderNumpy(self.config, self.params).forward(token_ids)

        if self.params is None:
            raise ValueError("Encoder parameters are not initialized")

        if not self._params_prepared:
            self._prepare_contiguous_params()

        cfg = self.config
        p = self.params

        token_ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        B = token_ids.shape[0]

        doc_h = p.embed[token_ids]
        mem = p.memory.expand_to_batch(B).astype(np.float32, copy=False)

        x = np.concatenate([doc_h, mem], axis=1)
        x = x + p.pos_embed[:x.shape[1]][None, :, :]
        x = np.ascontiguousarray(x, dtype=np.float32)

        doc_mask = token_ids != cfg.pad_id
        mem_mask = np.ones((B, cfg.n_memory_tokens), dtype=np.uint8)
        attention_mask = np.ascontiguousarray(
            np.concatenate([doc_mask.astype(np.uint8, copy=False), mem_mask], axis=1),
            dtype=np.uint8,
        )

        for layer in p.layers:
            x = self._ext.encoder_block_hybrid_blockwise(
                x,
                layer.Wq,
                layer.Wk,
                layer.Wv,
                layer.Wo,
                layer.W1,
                layer.W2,
                layer.norm1,
                layer.norm2,
                attention_mask,
                cfg.n_heads,
                cfg.head_dim,
                cfg.eps,
                self.num_threads,
                64,
            )

        x = _rms_norm(x, p.norm_final, cfg.eps)

        return np.ascontiguousarray(
            x[:, -cfg.n_memory_tokens:, :],
            dtype=np.float32,
        )

    def encode_retrieval(
        self,
        token_ids: np.ndarray,
        pooling: str = "mean",
    ) -> np.ndarray:
        mem_states = self.forward(token_ids)
        pooled = pool_memory(mem_states, pooling)

        self._last_pooled = pooled

        query = pooled @ self.retrieval_proj + self.query_bias
        query = np.ascontiguousarray(query, dtype=np.float32)

        return query

    def backward_query(
        self,
        token_ids: np.ndarray,
        grad_query: np.ndarray,
        lr: float = 1e-3,
    ) -> None:
        pooled = self._last_pooled
        if pooled is None:
            mem_states = self.forward(token_ids)
            pooled = pool_memory(mem_states, "mean")
            self._last_pooled = pooled

        B, H = pooled.shape

        grad_W = pooled.T @ grad_query / max(B, 1)
        b = grad_query.mean(axis=0)

        self.retrieval_proj -= lr * grad_W.astype(np.float32)
        self.query_bias -= lr * b.astype(np.float32)


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