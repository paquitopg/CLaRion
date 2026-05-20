from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
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

    h = config.hidden_dim
    f = config.ffn_dim
    scale = config.init_scale
    max_len = config.max_seq_len + config.n_memory_tokens

    def randn(*shape: int) -> np.ndarray:
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    layers: list[TransformerLayerParams] = []
    for _ in range(config.n_layers):
        layers.append(
            TransformerLayerParams(
                Wq=randn(h, h),
                Wk=randn(h, h),
                Wv=randn(h, h),
                Wo=randn(h, h),
                W1=randn(h, f),
                W2=randn(f, h),
                norm1=np.ones(h, dtype=np.float32),
                norm2=np.ones(h, dtype=np.float32),
            )
        )

    return EncoderParams(
        embed=randn(config.vocab_size, h),
        pos_embed=randn(max_len, h),
        layers=layers,
        norm_final=np.ones(h, dtype=np.float32),
        memory=MemoryTokens(config),
    )


def _rms_norm(x: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    """RMS normalization."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps, dtype=np.float32)
    return np.ascontiguousarray((x / rms) * scale, dtype=np.float32)


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi, dtype=np.float32) * (x + 0.044715 * x**3))
    )


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
    bsz, seq_len, hidden = x.shape

    q = x @ layer.Wq
    k = x @ layer.Wk
    v = x @ layer.Wv

    q = q.reshape(bsz, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim, dtype=np.float32)

    if attention_mask is not None:
        scores = np.where(attention_mask[:, None, None, :], scores, -1e9)

    weights = _softmax_lastdim(scores)
    context = weights @ v
    out = context.transpose(0, 2, 1, 3).reshape(bsz, seq_len, hidden)
    return np.ascontiguousarray(out @ layer.Wo, dtype=np.float32)


def _ffn_numpy(x: np.ndarray, layer: TransformerLayerParams) -> np.ndarray:
    """Feed-forward network."""
    return np.ascontiguousarray(_gelu(x @ layer.W1) @ layer.W2, dtype=np.float32)


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

        self.train_retrieval_head = True
        self.train_memory_tokens = True

        h = self.config.hidden_dim
        self.retrieval_proj = np.eye(h, dtype=np.float32)
        self.query_bias = np.zeros(h, dtype=np.float32)

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

    def _backward_query_numpy_impl(
        self,
        token_ids: np.ndarray,
        grad_query: np.ndarray,
        lr: float = 1e-3,
    ) -> None:
        pooled = getattr(self, "_last_pooled", None)
        if pooled is None:
            mem_states = self.forward(token_ids)
            pooled = pool_memory(mem_states, "mean")
            self._last_pooled = pooled

        grad_query = np.ascontiguousarray(grad_query, dtype=np.float32)

        bsz, hidden = pooled.shape
        n_mem = self.config.n_memory_tokens

        if self.train_retrieval_head:
            grad_w = pooled.T @ grad_query / max(bsz, 1)
            grad_b = grad_query.mean(axis=0)
            self.retrieval_proj -= lr * grad_w.astype(np.float32, copy=False)
            self.query_bias -= lr * grad_b.astype(np.float32, copy=False)

        if self.train_memory_tokens and n_mem > 0:
            grad_pooled = grad_query @ self.retrieval_proj.T
            grad_mem_states = np.broadcast_to(
                grad_pooled[:, None, :] / float(n_mem),
                (bsz, n_mem, hidden),
            ).copy()
            grad_memory = grad_mem_states.mean(axis=0)

            mem = self.params.memory.weights
            self.params.memory.weights = np.ascontiguousarray(
                mem - lr * grad_memory.astype(np.float32, copy=False),
                dtype=np.float32,
            )

    def backward_query(
        self,
        token_ids: np.ndarray,
        grad_query: np.ndarray,
        lr: float = 1e-3,
    ) -> None:
        """Update retrieval head and memory tokens from dL/dquery."""
        raise NotImplementedError


class EncoderNumpy(EncoderBackend):
    """Pure NumPy encoder implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_pooled: np.ndarray | None = None

    def _build_inputs_and_mask(self, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        p = self.params

        token_ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        bsz, seq_len = token_ids.shape

        doc_mask = token_ids != cfg.pad_id
        doc_h = p.embed[token_ids]
        mem = p.memory.expand_to_batch(bsz).astype(np.float32, copy=False)

        total_len = seq_len + cfg.n_memory_tokens
        x = np.empty((bsz, total_len, cfg.hidden_dim), dtype=np.float32)
        x[:, :seq_len, :] = doc_h
        x[:, seq_len:, :] = mem
        x += p.pos_embed[:total_len][None, :, :]

        mem_mask = np.ones((bsz, cfg.n_memory_tokens), dtype=bool)
        attention_mask = np.concatenate([doc_mask, mem_mask], axis=1)
        return x, attention_mask

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        cfg = self.config
        x, attention_mask = self._build_inputs_and_mask(token_ids)

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
        return np.ascontiguousarray(query, dtype=np.float32)

    def backward_query(
        self,
        token_ids: np.ndarray,
        grad_query: np.ndarray,
        lr: float = 1e-3,
    ) -> None:
        self._backward_query_numpy_impl(token_ids, grad_query, lr)


class EncoderCython(EncoderBackend):
    """Cython-backed encoder with NumPy fallback."""

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
        self._numpy_fallback: EncoderNumpy | None = None

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
        p.memory.weights = np.ascontiguousarray(p.memory.weights, dtype=np.float32)

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

    def _build_inputs_and_mask(self, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        p = self.params

        token_ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        bsz, seq_len = token_ids.shape

        doc_mask = token_ids != cfg.pad_id
        doc_h = p.embed[token_ids]
        mem = p.memory.expand_to_batch(bsz).astype(np.float32, copy=False)

        total_len = seq_len + cfg.n_memory_tokens
        x = np.empty((bsz, total_len, cfg.hidden_dim), dtype=np.float32)
        x[:, :seq_len, :] = doc_h
        x[:, seq_len:, :] = mem
        x += p.pos_embed[:total_len][None, :, :]

        mem_mask = np.ones((bsz, cfg.n_memory_tokens), dtype=np.uint8)
        attention_mask = np.ascontiguousarray(
            np.concatenate([doc_mask.astype(np.uint8, copy=False), mem_mask], axis=1),
            dtype=np.uint8,
        )
        return x, attention_mask

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if not self._available:
            if self._numpy_fallback is None:
                self._numpy_fallback = EncoderNumpy(self.config, self.params)
            return self._numpy_fallback.forward(token_ids)

        if self.params is None:
            raise ValueError("Encoder parameters are not initialized")

        if not self._params_prepared:
            self._prepare_contiguous_params()

        cfg = self.config
        p = self.params

        x, attention_mask = self._build_inputs_and_mask(token_ids)

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
        return np.ascontiguousarray(query, dtype=np.float32)

    def backward_query(
        self,
        token_ids: np.ndarray,
        grad_query: np.ndarray,
        lr: float = 1e-3,
    ) -> None:
        self._backward_query_numpy_impl(token_ids, grad_query, lr)


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


def save_encoder_weights(encoder: EncoderBackend, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    p = encoder.params
    arrays: dict[str, np.ndarray] = {
        "embed": np.asarray(p.embed, dtype=np.float32),
        "pos_embed": np.asarray(p.pos_embed, dtype=np.float32),
        "norm_final": np.asarray(p.norm_final, dtype=np.float32),
        "memory": np.asarray(p.memory.weights, dtype=np.float32),
        "retrieval_proj": np.asarray(encoder.retrieval_proj, dtype=np.float32),
        "query_bias": np.asarray(encoder.query_bias, dtype=np.float32),
        "n_layers": np.asarray([len(p.layers)], dtype=np.int32),
    }

    for i, layer in enumerate(p.layers):
        arrays[f"layers.{i}.Wq"] = np.asarray(layer.Wq, dtype=np.float32)
        arrays[f"layers.{i}.Wk"] = np.asarray(layer.Wk, dtype=np.float32)
        arrays[f"layers.{i}.Wv"] = np.asarray(layer.Wv, dtype=np.float32)
        arrays[f"layers.{i}.Wo"] = np.asarray(layer.Wo, dtype=np.float32)
        arrays[f"layers.{i}.W1"] = np.asarray(layer.W1, dtype=np.float32)
        arrays[f"layers.{i}.W2"] = np.asarray(layer.W2, dtype=np.float32)
        arrays[f"layers.{i}.norm1"] = np.asarray(layer.norm1, dtype=np.float32)
        arrays[f"layers.{i}.norm2"] = np.asarray(layer.norm2, dtype=np.float32)

    np.savez_compressed(path, **arrays)


def load_encoder_weights(encoder: EncoderBackend, path: str) -> None:
    ckpt = np.load(path)
    p = encoder.params

    p.embed[...] = ckpt["embed"]
    p.pos_embed[...] = ckpt["pos_embed"]
    p.norm_final[...] = ckpt["norm_final"]
    p.memory.weights[...] = ckpt["memory"]

    encoder.retrieval_proj[...] = ckpt["retrieval_proj"]
    encoder.query_bias[...] = ckpt["query_bias"]

    n_layers = int(ckpt["n_layers"][0])
    if n_layers != len(p.layers):
        raise ValueError(f"Layer mismatch: checkpoint={n_layers}, model={len(p.layers)}")

    for i, layer in enumerate(p.layers):
        layer.Wq[...] = ckpt[f"layers.{i}.Wq"]
        layer.Wk[...] = ckpt[f"layers.{i}.Wk"]
        layer.Wv[...] = ckpt[f"layers.{i}.Wv"]
        layer.Wo[...] = ckpt[f"layers.{i}.Wo"]
        layer.W1[...] = ckpt[f"layers.{i}.W1"]
        layer.W2[...] = ckpt[f"layers.{i}.W2"]
        layer.norm1[...] = ckpt[f"layers.{i}.norm1"]
        layer.norm2[...] = ckpt[f"layers.{i}.norm2"]

    if isinstance(encoder, EncoderCython):
        encoder._params_prepared = False
        encoder._prepare_contiguous_params()