from __future__ import annotations

import numpy as np
from typing import Optional

from .config import DecoderConfig


class DecoderBackend:

    def __init__(self, config: DecoderConfig, params=None):
        self.config = config
        self.params = params

    def forward(
        self,
        input_ids: np.ndarray,
        memory: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def backward(
        self,
        grad_logits: np.ndarray,
        lr: float = 1e-3,
    ):
        raise NotImplementedError

    def generate(
        self,
        input_ids: np.ndarray,
        memory: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        topk: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> np.ndarray:

        generated = input_ids

        for _ in range(max_new_tokens):

            logits = self.forward(generated, memory)

            logits = logits[:, -1, :] / temperature

            logits = logits - np.max(logits, axis=-1, keepdims=True)

            probs = np.exp(logits)
            probs /= np.sum(probs, axis=-1, keepdims=True)

            if topk > 0:

                idx = np.argpartition(
                    probs,
                    -topk,
                    axis=-1,
                )[..., -topk:]

                top_probs = np.take_along_axis(
                    probs,
                    idx,
                    axis=-1,
                )

                top_probs /= np.sum(
                    top_probs,
                    axis=-1,
                    keepdims=True,
                )

                next_token = np.array([
                    np.random.choice(idx[i], p=top_probs[i])
                    for i in range(probs.shape[0])
                ], dtype=np.int64)

            else:

                next_token = np.array([
                    np.random.choice(
                        probs.shape[-1],
                        p=probs[i],
                    )
                    for i in range(probs.shape[0])
                ], dtype=np.int64)

            next_token = next_token[:, None]

            generated = np.concatenate(
                [generated, next_token],
                axis=1,
            )

            if (
                eos_token_id is not None
                and np.all(next_token == eos_token_id)
            ):
                break

        return generated


def init_decoder_weights(cfg):

    rng = np.random.default_rng(42)

    def randn(*shape):
        return (
            rng.normal(
                0.0,
                cfg.init_scale,
                size=shape,
            ).astype(np.float32)
        )

    return {
        "embed": randn(cfg.vocab_size, cfg.hidden_dim),

        "mem_proj": randn(
            cfg.hidden_dim,
            cfg.hidden_dim,
        ),

        "layers": [
            {
                "Wq": randn(cfg.hidden_dim, cfg.hidden_dim),
                "Wk": randn(cfg.hidden_dim, cfg.hidden_dim),
                "Wv": randn(cfg.hidden_dim, cfg.hidden_dim),
                "Wo": randn(cfg.hidden_dim, cfg.hidden_dim),

                "W1": randn(cfg.hidden_dim, cfg.ffn_dim),
                "W2": randn(cfg.ffn_dim, cfg.hidden_dim),

                "norm1": np.ones(
                    cfg.hidden_dim,
                    dtype=np.float32,
                ),

                "norm2": np.ones(
                    cfg.hidden_dim,
                    dtype=np.float32,
                ),
            }
            for _ in range(cfg.n_layers)
        ],

        "lm_head": randn(
            cfg.hidden_dim,
            cfg.vocab_size,
        ),
    }


def _rms_norm(x, scale, eps=1e-6):

    rms = np.sqrt(
        np.mean(x * x, axis=-1, keepdims=True) + eps
    )

    return (x / rms) * scale


def _softmax(x):

    x = x - np.max(x, axis=-1, keepdims=True)

    e = np.exp(x)

    return e / np.sum(e, axis=-1, keepdims=True)


def _gelu(x):

    return (
        0.5
        * x
        * (
            1.0
            + np.tanh(
                np.sqrt(2.0 / np.pi)
                * (
                    x
                    + 0.044715 * x**3
                )
            )
        )
    )


def _cross_attention(
    x,
    memory,
    Wq,
    Wk,
    Wv,
    Wo,
    n_heads,
    head_dim,
):

    B, T, H = x.shape
    S = memory.shape[1]

    Q = x @ Wq
    K = memory @ Wk
    V = memory @ Wv

    Q = Q.reshape(
        B,
        T,
        n_heads,
        head_dim,
    ).transpose(0, 2, 1, 3)

    K = K.reshape(
        B,
        S,
        n_heads,
        head_dim,
    ).transpose(0, 2, 1, 3)

    V = V.reshape(
        B,
        S,
        n_heads,
        head_dim,
    ).transpose(0, 2, 1, 3)

    scores = (
        Q @ K.transpose(0, 1, 3, 2)
    ) / np.sqrt(head_dim)

    weights = _softmax(scores)

    ctx = weights @ V

    ctx = ctx.transpose(
        0,
        2,
        1,
        3,
    ).reshape(B, T, H)

    return ctx @ Wo


def _ffn(x, W1, W2):

    return _gelu(x @ W1) @ W2


class DecoderNumpy(DecoderBackend):

    def __init__(self, config, params=None):

        super().__init__(config)

        self.params = (
            params
            if params is not None
            else init_decoder_weights(config)
        )

    def forward(
        self,
        input_ids,
        memory,
    ):

        cfg = self.config
        weights = self.params

        input_ids = np.asarray(
            input_ids,
            dtype=np.int64,
        )

        memory = np.asarray(
            memory,
            dtype=np.float32,
        )

        x = weights["embed"][input_ids]

        memory = memory @ weights["mem_proj"]

        for layer in weights["layers"]:

            h = _rms_norm(
                x,
                layer["norm1"],
                cfg.eps,
            )

            x = x + _cross_attention(
                h,
                memory,
                layer["Wq"],
                layer["Wk"],
                layer["Wv"],
                layer["Wo"],
                cfg.n_heads,
                cfg.hidden_dim // cfg.n_heads,
            )

            h = _rms_norm(
                x,
                layer["norm2"],
                cfg.eps,
            )

            x = x + _ffn(
                h,
                layer["W1"],
                layer["W2"],
            )

        x = _rms_norm(
            x,
            np.ones(
                x.shape[-1],
                dtype=np.float32,
            ),
            cfg.eps,
        )

        self.last_hidden = x

        logits = x @ weights["lm_head"]

        return logits.astype(np.float32)

    def backward(
        self,
        grad_logits,
        lr=1e-3,
    ):

        weights = self.params

        x = self.last_hidden[:, :-1, :]

        B, T, H = x.shape
        V = grad_logits.shape[-1]

        x_flat = x.reshape(-1, H)

        g_flat = grad_logits.reshape(-1, V)

        grad_lm_head = x_flat.T @ g_flat

        weights["lm_head"] -= (
            lr
            * grad_lm_head.astype(np.float32)
        )


class DecoderCython(DecoderBackend):
    def __init__(self, config, params=None, num_threads=0):
        super().__init__(config, params)
        self.params = params if params is not None else init_decoder_weights(config)
        self.num_threads = num_threads

        try:
            from src.parallel import cython_decoder
            self._ext = cython_decoder
            self._available = True
        except Exception:
            self._ext = None
            self._available = False

    def forward(self, input_ids, memory):
        if not self._available:
            return DecoderNumpy(self.config, self.params).forward(input_ids, memory)

        cfg = self.config
        w = self.params

        input_ids = np.asarray(input_ids, dtype=np.int64)
        memory = np.asarray(memory, dtype=np.float32)

        x = np.ascontiguousarray(w["embed"][input_ids], dtype=np.float32)
        mem = np.ascontiguousarray(memory @ w["mem_proj"], dtype=np.float32)

        for layer in w["layers"]:
            x = self._ext.decoder_forward_cython(
                x,
                mem,
                np.ascontiguousarray(layer["Wq"], dtype=np.float32),
                np.ascontiguousarray(layer["Wk"], dtype=np.float32),
                np.ascontiguousarray(layer["Wv"], dtype=np.float32),
                np.ascontiguousarray(layer["Wo"], dtype=np.float32),
                np.ascontiguousarray(layer["W1"], dtype=np.float32),
                np.ascontiguousarray(layer["W2"], dtype=np.float32),
                np.ascontiguousarray(layer["norm1"], dtype=np.float32),
                np.ascontiguousarray(layer["norm2"], dtype=np.float32),
                cfg.eps,
            )

        x = _rms_norm(
            x,
            np.ones(x.shape[-1], dtype=np.float32),
            cfg.eps,
        )

        self.last_hidden = x
        logits = self._ext.project_lm_head(
            x,
            np.ascontiguousarray(w["lm_head"], dtype=np.float32),
        )
        return logits.astype(np.float32)

    def backward(self, grad_logits, lr=1e-3):
        if not self._available:
            return DecoderNumpy(self.config, self.params).backward(grad_logits, lr)

        self._ext.decoder_backward_lm_head(
            np.ascontiguousarray(self.last_hidden[:, :-1, :], dtype=np.float32),
            np.ascontiguousarray(grad_logits, dtype=np.float32),
            np.ascontiguousarray(self.params["lm_head"], dtype=np.float32),
            lr,
        )

def build_decoder(
    config,
    backend="numpy",
    **kwargs,
):

    if backend == "numpy":
        return DecoderNumpy(config)

    if backend == "cython":
        return DecoderCython(
            config,
            **kwargs,
        )

    raise ValueError(
        f"Unknown backend: {backend}"
    )