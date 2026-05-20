from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

from .config import LossConfig

logger = logging.getLogger(__name__)


def _softmax_python(vec):
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


def _loss_python(logits, labels, ignore_index=0):
    B = len(logits)
    loss_sum = 0.0
    count = 0

    for b in range(B):
        for t in range(len(logits[b]) - 1):
            y = labels[b][t + 1]
            if y == ignore_index:
                continue

            probs = _softmax_python(logits[b][t])
            loss_sum += -math.log(probs[y] + 1e-12)
            count += 1

    return loss_sum / max(count, 1)


def _loss_numpy(logits: np.ndarray, labels: np.ndarray, ignore_index=0):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    B, T, V = logits.shape
    x = logits.reshape(-1, V)
    y = labels.reshape(-1)

    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    idx = np.arange(len(y))
    safe_y = np.where(y == ignore_index, 0, y)
    log_probs = -np.log(probs[idx, safe_y] + 1e-12)

    mask = (y != ignore_index)
    return np.sum(log_probs * mask) / (np.sum(mask) + 1e-12)


def _loss_cython(
    logits: np.ndarray,
    labels: np.ndarray,
    ignore_index: int = 0,
    num_threads: int = 0,
):
    try:
        from src.parallel import cython_loss
    except Exception as e:
        logger.warning("Cython loss unavailable: %s", e)
        return _loss_numpy(logits, labels, ignore_index)

    logits = np.ascontiguousarray(logits, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int32)

    return cython_loss.clara_lm_loss_cython_fast(
        logits,
        labels,
        pad_id=ignore_index,
        num_threads=num_threads,
    )


def clara_lm_loss(
    logits,
    labels,
    config: Optional[LossConfig] = None,
    backend: str = "numpy",
):
    config = config or LossConfig()

    if backend == "numpy":
        return _loss_numpy(logits, labels, config.ignore_index)

    if backend == "python":
        return _loss_python(logits, labels, config.ignore_index)

    if backend == "cython":
        return _loss_cython(
            logits,
            labels,
            config.ignore_index,
            config.num_threads,
        )

    raise ValueError(f"Unknown backend: {backend}")


def cross_entropy_with_grad(
    logits,
    targets,
    ignore_index=0,
    backend: str = "numpy",
):
    if backend == "cython":
        try:
            from src.parallel import cython_loss
            logits = np.ascontiguousarray(logits, dtype=np.float32)
            targets = np.ascontiguousarray(targets, dtype=np.int32)
            return cython_loss.clara_ce_with_grad_cython(
                logits,
                targets,
                ignore_index,
            )
        except Exception as e:
            logger.warning("Cython fused loss unavailable: %s", e)

    x = logits[:, :-1, :]
    y = targets[:, 1:]

    B, T, V = x.shape

    x_flat = x.reshape(-1, V)
    y_flat = y.reshape(-1)

    mask = (y_flat != ignore_index)

    x_shift = x_flat - np.max(x_flat, axis=1, keepdims=True)
    exp = np.exp(x_shift)
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    idx = np.arange(len(y_flat))
    safe_y = np.where(y_flat == ignore_index, 0, y_flat)

    loss_vec = -np.log(probs[idx, safe_y] + 1e-12)
    loss = (loss_vec * mask).sum() / (mask.sum() + 1e-12)

    grad = probs.copy()
    grad[idx, safe_y] -= 1.0
    grad *= mask[:, None]
    grad /= (mask.sum() + 1e-12)
    grad = grad.reshape(B, T, V)

    return loss, grad