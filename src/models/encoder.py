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


# --------------------------------------------------------------------------- #
# Parameter container
# --------------------------------------------------------------------------- #
@dataclass
class TransformerLayerParams:
    """One transformer block's parameters, all float32."""
    # Attention
    Wq: np.ndarray  # (H, H)
    Wk: np.ndarray  # (H, H)
    Wv: np.ndarray  # (H, H)
    Wo: np.ndarray  # (H, H)
    # FFN
    W1: np.ndarray  # (H, F)
    W2: np.ndarray  # (F, H)
    # Norms (pre-norm RMSNorm scale)
    norm1: np.ndarray  # (H,)
    norm2: np.ndarray  # (H,)


@dataclass
class EncoderParams:
    embed: np.ndarray            # (V, H) token embedding table
    layers: list[TransformerLayerParams]
    norm_final: np.ndarray       # (H,) final RMSNorm scale
    memory: MemoryTokens


def _init_params(config: ModelConfig) -> EncoderParams:
    """Initialize encoder parameters from a deterministic RNG."""
    rng = np.random.default_rng(config.rng_seed)
    scale = config.init_scale
    H = config.hidden_dim
    F = config.ffn_dim

    def randn(*shape):
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    layers: list[TransformerLayerParams] = []
    for _ in range(config.n_layers):
        layers.append(TransformerLayerParams(
            Wq=randn(H, H), Wk=randn(H, H), Wv=randn(H, H), Wo=randn(H, H),
            W1=randn(H, F), W2=randn(F, H),
            norm1=np.ones(H, dtype=np.float32),
            norm2=np.ones(H, dtype=np.float32),
        ))

    return EncoderParams(
        embed=randn(config.vocab_size, H),
        layers=layers,
        norm_final=np.ones(H, dtype=np.float32),
        memory=MemoryTokens(config),
    )


# --------------------------------------------------------------------------- #
# Numpy reference implementation
# --------------------------------------------------------------------------- #
def _rms_norm(x: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    """Pre-norm RMSNorm: x * scale / rms(x). Broadcasts across the last dim."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


def _gelu(x: np.ndarray) -> np.ndarray:
    """Approximate GELU (the variant most transformers ship)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax_lastdim(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over the last dimension."""
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attention_numpy(
    x: np.ndarray,
    layer: TransformerLayerParams,
    n_heads: int,
    head_dim: int,
) -> np.ndarray:
    """
    Bidirectional (non-causal) multi-head self-attention.

    Bidirectional is correct here: the compressor is an *encoder*, every token
    (including memory tokens) can see every other token. This matches the
    CLaRa compressor in the paper.
    """
    B, L, H = x.shape

    Q = x @ layer.Wq  # (B, L, H)
    K = x @ layer.Wk
    V = x @ layer.Wv

    # Reshape into heads: (B, n_heads, L, head_dim)
    Q = Q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)

    scale = 1.0 / np.sqrt(head_dim)
    scores = (Q @ K.transpose(0, 1, 3, 2)) * scale          # (B, h, L, L)
    weights = _softmax_lastdim(scores)
    context = weights @ V                                   # (B, h, L, head_dim)

    out = context.transpose(0, 2, 1, 3).reshape(B, L, H)    # (B, L, H)
    return out @ layer.Wo


def _ffn_numpy(x: np.ndarray, layer: TransformerLayerParams) -> np.ndarray:
    return _gelu(x @ layer.W1) @ layer.W2


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class EncoderBackend:
    """Abstract backend signature. Concrete classes implement `forward`."""
    name: str = "abstract"

    def __init__(self, config: ModelConfig, params: Optional[EncoderParams] = None):
        self.config = config
        self.params = params if params is not None else _init_params(config)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Encode a batch of documents into memory-token embeddings.

        Args:
            token_ids: (B, L_doc) int32 array. Use pad_id for padding.

        Returns:
            (B, n_memory_tokens * hidden_dim) float32 doc embeddings, flat.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Pure-numpy backend (baseline)
# --------------------------------------------------------------------------- #
class EncoderNumpy(EncoderBackend):
    """Pure-numpy reference. Baseline for the CPU-parallelization study."""
    name = "numpy"

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        cfg = self.config
        assert token_ids.dtype in (np.int32, np.int64), "token_ids must be integer"
        B, L_doc = token_ids.shape

        # 1) Token embedding lookup.
        doc_h = self.params.embed[token_ids].astype(np.float32, copy=False)  # (B, L_doc, H)

        # 2) Append memory tokens at the tail. Broadcast across batch.
        mem = self.params.memory.expand_to_batch(B)                          # (B, l, H)
        x = np.concatenate([doc_h, mem], axis=1)                             # (B, L_doc+l, H)

        # 3) Transformer stack (pre-norm).
        for layer in self.params.layers:
            x = x + _attention_numpy(_rms_norm(x, layer.norm1, cfg.eps),
                                     layer, cfg.n_heads, cfg.head_dim)
            x = x + _ffn_numpy(_rms_norm(x, layer.norm2, cfg.eps), layer)

        # 4) Final norm, then slice out the memory-token tail.
        x = _rms_norm(x, self.params.norm_final, cfg.eps)
        mem_states = x[:, -cfg.n_memory_tokens:, :]                          # (B, l, H)

        # 5) Flatten into doc embeddings: (B, l*H).
        return mem_states.reshape(B, cfg.embedding_dim).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Cython + OpenMP backend
# --------------------------------------------------------------------------- #
class EncoderCython(EncoderBackend):
    """
    Delegates the per-layer hot kernels to the OpenMP-parallel Cython module.

    Falls back to the numpy reference at construction time if the extension
    has not been built, so importing this module never raises.
    """
    name = "cython"

    def __init__(self, config: ModelConfig, params: Optional[EncoderParams] = None,
                 num_threads: int = 0):
        super().__init__(config, params)
        self.num_threads = num_threads
        try:
            from src.parallel import cython_encoder  # type: ignore
            self._ext = cython_encoder
            self._available = True
            logger.info("Cython encoder backend loaded (OpenMP).")
        except Exception as e:
            self._ext = None
            self._available = False
            logger.warning("Cython encoder unavailable, will fall back to numpy: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if not self._available:
            return EncoderNumpy(self.config, self.params).forward(token_ids)

        cfg = self.config
        B, L_doc = token_ids.shape

        # Stage 1 (embed + memory-token append) stays in numpy: pure gather, cheap.
        doc_h = self.params.embed[token_ids].astype(np.float32, copy=False)
        mem = self.params.memory.expand_to_batch(B)
        x = np.ascontiguousarray(np.concatenate([doc_h, mem], axis=1), dtype=np.float32)

        # Stage 2: each transformer block, hot kernels in Cython.
        for layer in self.params.layers:
            x = self._ext.encoder_block(
                x,
                layer.Wq, layer.Wk, layer.Wv, layer.Wo,
                layer.W1, layer.W2,
                layer.norm1, layer.norm2,
                cfg.n_heads, cfg.head_dim, cfg.eps,
                self.num_threads,
            )

        # Final norm + memory-token slice — cheap, stays in numpy.
        x = _rms_norm(x, self.params.norm_final, cfg.eps)
        mem_states = x[:, -cfg.n_memory_tokens:, :]
        return mem_states.reshape(B, cfg.embedding_dim).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Public factory
# --------------------------------------------------------------------------- #
def build_encoder(
    config: ModelConfig,
    backend: str = "numpy",
    num_threads: int = 0,
    params: Optional[EncoderParams] = None,
) -> EncoderBackend:
    """Construct an encoder backend by name.

    Sharing `params` across backends is the only way to make naive-vs-fast
    benchmarks apples-to-apples — same weights, same input, just different
    compute paths.
    """
    if backend == "numpy":
        return EncoderNumpy(config, params)
    if backend == "cython":
        return EncoderCython(config, params, num_threads=num_threads)
    raise ValueError(f"Unknown encoder backend: {backend!r}")
