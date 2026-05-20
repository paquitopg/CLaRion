from __future__ import annotations

import numpy as np
from typing import Optional

from .config import DecoderConfig


class DecoderBackend:
    def __init__(self, config: DecoderConfig, params=None):
        self.config = config
        self.params = params
        self.last_hidden: np.ndarray | None = None
        self.last_memory: np.ndarray | None = None
        self.last_mem_proj: np.ndarray | None = None

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
        return_grad_memory: bool = False,
        update_mem_proj: bool = False,
    ) -> np.ndarray | None:
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
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = logits - np.max(logits, axis=-1, keepdims=True)

            probs = np.exp(logits)
            probs /= np.sum(probs, axis=-1, keepdims=True)

            if topk > 0:
                idx = np.argpartition(probs, -topk, axis=-1)[..., -topk:]
                top_probs = np.take_along_axis(probs, idx, axis=-1)
                top_probs /= np.sum(top_probs, axis=-1, keepdims=True)

                next_token = np.array(
                    [np.random.choice(idx[i], p=top_probs[i]) for i in range(probs.shape[0])],
                    dtype=np.int64,
                )
            else:
                next_token = np.array(
                    [np.random.choice(probs.shape[-1], p=probs[i]) for i in range(probs.shape[0])],
                    dtype=np.int64,
                )

            next_token = next_token[:, None]
            generated = np.concatenate([generated, next_token], axis=1)

            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

        return generated


def init_decoder_weights(cfg: DecoderConfig):
    rng = np.random.default_rng(42)

    def randn(*shape):
        return np.ascontiguousarray(
            rng.normal(0.0, cfg.init_scale, size=shape).astype(np.float32)
        )

    return {
        "embed": randn(cfg.vocab_size, cfg.hidden_dim),
        "mem_proj": randn(cfg.hidden_dim, cfg.hidden_dim),
        "layers": [
            {
                "Wq": randn(cfg.hidden_dim, cfg.hidden_dim),
                "Wk": randn(cfg.hidden_dim, cfg.hidden_dim),
                "Wv": randn(cfg.hidden_dim, cfg.hidden_dim),
                "Wo": randn(cfg.hidden_dim, cfg.hidden_dim),
                "W1": randn(cfg.hidden_dim, cfg.ffn_dim),
                "W2": randn(cfg.ffn_dim, cfg.hidden_dim),
                "norm1": np.ascontiguousarray(np.ones(cfg.hidden_dim, dtype=np.float32)),
                "norm2": np.ascontiguousarray(np.ones(cfg.hidden_dim, dtype=np.float32)),
            }
            for _ in range(cfg.n_layers)
        ],
        "lm_head": randn(cfg.hidden_dim, cfg.vocab_size),
    }


def _rms_norm(x, scale, eps=1e-6):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _gelu(x):
    return 0.5 * x * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
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

    Q = Q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    weights = _softmax(scores)
    ctx = weights @ V
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, H)

    return ctx @ Wo


def _ffn(x, W1, W2):
    return _gelu(x @ W1) @ W2


class DecoderNumpy(DecoderBackend):
    def __init__(self, config, params=None):
        super().__init__(config, params)
        self.params = params if params is not None else init_decoder_weights(config)

    def forward(self, input_ids, memory):
        cfg = self.config
        weights = self.params

        input_ids = np.asarray(input_ids, dtype=np.int64)
        memory = np.asarray(memory, dtype=np.float32)

        x = weights["embed"][input_ids]
        mem = np.ascontiguousarray(memory @ weights["mem_proj"], dtype=np.float32)

        self.last_memory = np.ascontiguousarray(memory, dtype=np.float32)
        self.last_mem_proj = mem

        for layer in weights["layers"]:
            h = _rms_norm(x, layer["norm1"], cfg.eps)
            x = x + _cross_attention(
                h,
                mem,
                layer["Wq"],
                layer["Wk"],
                layer["Wv"],
                layer["Wo"],
                cfg.n_heads,
                cfg.hidden_dim // cfg.n_heads,
            )

            h = _rms_norm(x, layer["norm2"], cfg.eps)
            x = x + _ffn(h, layer["W1"], layer["W2"])

        x = _rms_norm(
            x,
            np.ones(x.shape[-1], dtype=np.float32),
            cfg.eps,
        )

        self.last_hidden = np.ascontiguousarray(x, dtype=np.float32)
        logits = x @ weights["lm_head"]
        return np.ascontiguousarray(logits, dtype=np.float32)

    def backward(
        self,
        grad_logits,
        lr=1e-3,
        return_grad_memory=False,
        update_mem_proj=False,
    ):
        if self.last_hidden is None:
            raise RuntimeError("forward() must be called before backward().")

        weights = self.params
        grad_logits = np.ascontiguousarray(grad_logits, dtype=np.float32)
        hidden = np.ascontiguousarray(self.last_hidden[:, :-1, :], dtype=np.float32)

        if grad_logits.ndim != 3:
            raise ValueError(f"grad_logits must be 3D, got {grad_logits.shape}")
        if grad_logits.shape[:2] != hidden.shape[:2]:
            raise ValueError(f"Mismatch hidden={hidden.shape} vs grad_logits={grad_logits.shape}")

        B, T, H = hidden.shape
        V = grad_logits.shape[-1]

        hidden_flat = hidden.reshape(-1, H)
        grad_flat = grad_logits.reshape(-1, V)

        lm_head_before = np.ascontiguousarray(weights["lm_head"], dtype=np.float32)

        grad_lm_head = np.ascontiguousarray(hidden_flat.T @ grad_flat, dtype=np.float32)
        grad_hidden = np.ascontiguousarray(grad_flat @ lm_head_before.T, dtype=np.float32)
        grad_hidden = grad_hidden.reshape(B, T, H)

        weights["lm_head"] -= lr * grad_lm_head

        if not return_grad_memory:
            return None

        if self.last_memory is None:
            raise RuntimeError("forward() must cache memory before backward().")

        last_memory = np.ascontiguousarray(self.last_memory, dtype=np.float32)
        S = last_memory.shape[1]

        grad_mem_proj_shared = np.ascontiguousarray(grad_hidden.mean(axis=1), dtype=np.float32)
        grad_memory_shared = np.ascontiguousarray(
            grad_mem_proj_shared @ weights["mem_proj"].T, dtype=np.float32
        )

        if update_mem_proj:
            grad_mem_proj_w = np.zeros((H, H), dtype=np.float32)
            for b in range(B):
                for s in range(S):
                    grad_mem_proj_w += np.outer(last_memory[b, s], grad_mem_proj_shared[b])
            grad_mem_proj_w /= max(B * S, 1)
            weights["mem_proj"] -= lr * np.ascontiguousarray(grad_mem_proj_w, dtype=np.float32)

        grad_memory = np.empty((B, S, H), dtype=np.float32)
        grad_memory[:] = grad_memory_shared[:, None, :]

        return grad_memory


class DecoderCython(DecoderBackend):
    def __init__(self, config, params=None, num_threads=0):
        super().__init__(config, params)

        self.params = params if params is not None else init_decoder_weights(config)
        self.num_threads = num_threads

        self._make_params_contiguous()

        try:
            from src.parallel import cython_decoder
            self._ext = cython_decoder
            self._available = True
        except Exception:
            self._ext = None
            self._available = False

    def _make_params_contiguous(self):
        w = self.params

        w["embed"] = np.ascontiguousarray(w["embed"], dtype=np.float32)
        w["mem_proj"] = np.ascontiguousarray(w["mem_proj"], dtype=np.float32)
        w["lm_head"] = np.ascontiguousarray(w["lm_head"], dtype=np.float32)

        for layer in w["layers"]:
            layer["Wq"] = np.ascontiguousarray(layer["Wq"], dtype=np.float32)
            layer["Wk"] = np.ascontiguousarray(layer["Wk"], dtype=np.float32)
            layer["Wv"] = np.ascontiguousarray(layer["Wv"], dtype=np.float32)
            layer["Wo"] = np.ascontiguousarray(layer["Wo"], dtype=np.float32)
            layer["W1"] = np.ascontiguousarray(layer["W1"], dtype=np.float32)
            layer["W2"] = np.ascontiguousarray(layer["W2"], dtype=np.float32)
            layer["norm1"] = np.ascontiguousarray(layer["norm1"], dtype=np.float32)
            layer["norm2"] = np.ascontiguousarray(layer["norm2"], dtype=np.float32)

    def forward(self, input_ids, memory):
        if not self._available:
            return DecoderNumpy(self.config, self.params).forward(input_ids, memory)

        cfg = self.config
        w = self.params

        input_ids = np.asarray(input_ids, dtype=np.int64)
        memory = np.asarray(memory, dtype=np.float32)

        x = w["embed"][input_ids]
        if not x.flags["C_CONTIGUOUS"] or x.dtype != np.float32:
            x = np.ascontiguousarray(x, dtype=np.float32)

        mem = memory @ w["mem_proj"]
        if not mem.flags["C_CONTIGUOUS"] or mem.dtype != np.float32:
            mem = np.ascontiguousarray(mem, dtype=np.float32)

        self.last_memory = memory if (memory.flags["C_CONTIGUOUS"] and memory.dtype == np.float32) else np.ascontiguousarray(memory, dtype=np.float32)
        self.last_mem_proj = mem

        for layer in w["layers"]:
            x = self._ext.decoder_forward_cython(
                x,
                mem,
                layer["Wq"],
                layer["Wk"],
                layer["Wv"],
                layer["Wo"],
                layer["W1"],
                layer["W2"],
                layer["norm1"],
                layer["norm2"],
                cfg.eps,
            )

        x = _rms_norm(
            x,
            np.ones(x.shape[-1], dtype=np.float32),
            cfg.eps,
        )

        self.last_hidden = x if (x.flags["C_CONTIGUOUS"] and x.dtype == np.float32) else np.ascontiguousarray(x, dtype=np.float32)
        logits = self._ext.project_lm_head(self.last_hidden, w["lm_head"])
        return logits if (logits.flags["C_CONTIGUOUS"] and logits.dtype == np.float32) else np.ascontiguousarray(logits, dtype=np.float32)

    def backward(
        self,
        grad_logits,
        lr=1e-3,
        return_grad_memory=False,
        update_mem_proj=False,
    ):
        if not self._available:
            return DecoderNumpy(self.config, self.params).backward(
                grad_logits,
                lr=lr,
                return_grad_memory=return_grad_memory,
                update_mem_proj=update_mem_proj,
            )

        if self.last_hidden is None:
            raise RuntimeError("forward() must be called before backward().")
        if self.last_memory is None and return_grad_memory:
            raise RuntimeError("forward() must cache memory before backward().")

        weights = self.params
        grad_logits = np.ascontiguousarray(grad_logits, dtype=np.float32)
        hidden = np.ascontiguousarray(self.last_hidden[:, :-1, :], dtype=np.float32)

        if grad_logits.shape[:2] != hidden.shape[:2]:
            raise ValueError(
                f"Mismatch hidden={hidden.shape} vs grad_logits={grad_logits.shape}"
            )

        B, T, H = hidden.shape
        V = grad_logits.shape[-1]
        S = self.last_memory.shape[1]

        lm_head_before = np.ascontiguousarray(weights["lm_head"], dtype=np.float32).copy()

        grad_flat = grad_logits.reshape(-1, V)
        grad_hidden = np.ascontiguousarray(grad_flat @ lm_head_before.T, dtype=np.float32)
        grad_hidden = grad_hidden.reshape(B, T, H)

        grad_mem_proj_shared = np.ascontiguousarray(
            grad_hidden.mean(axis=1), dtype=np.float32
        )
        grad_memory_shared = np.ascontiguousarray(
            grad_mem_proj_shared @ weights["mem_proj"].T, dtype=np.float32
        )

        if update_mem_proj:
            grad_mem_proj_full = np.broadcast_to(
                grad_mem_proj_shared[:, None, :],
                (B, S, H),
            )
            grad_mem_proj_w = (
                self.last_memory.reshape(-1, H).T @ grad_mem_proj_full.reshape(-1, H)
            ) / max(B * S, 1)
            weights["mem_proj"] -= lr * np.ascontiguousarray(grad_mem_proj_w, dtype=np.float32)

        grad_memory = np.broadcast_to(
            grad_memory_shared[:, None, :],
            (B, S, H),
        ).copy()

        return grad_memory


def build_decoder(config, backend="numpy", **kwargs):
    params = kwargs.get("params", None)

    if backend == "numpy":
        return DecoderNumpy(config, params=params)

    if backend == "cython":
        return DecoderCython(
            config,
            params=params,
            num_threads=kwargs.get("num_threads", 0),
        )

    raise ValueError(f"Unknown backend: {backend}")