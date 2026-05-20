from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import TopKConfig

logger = logging.getLogger("clara_topk")


@dataclass
class TopKResult:
    hard: np.ndarray      # (B, N) k-hot mask
    soft: np.ndarray      # (B, N) soft distribution / relaxation
    indices: np.ndarray   # (B, K)
    scores: np.ndarray    # (B, N)


class TopKBackend:
    def __init__(self, config: TopKConfig):
        self.config = config

    def forward(self, logits: np.ndarray) -> TopKResult:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray, cache: TopKResult) -> np.ndarray:
        raise NotImplementedError

    def benchmark(self, logits: np.ndarray, iters: int = 50) -> float:
        for _ in range(5):
            self.forward(logits)

        start = time.perf_counter()
        for _ in range(iters):
            self.forward(logits)
        end = time.perf_counter()

        return (end - start) / iters

class TopKNaive(TopKBackend):
    """
    Pure Python greedy Top-K (reference slow implementation).
    """

    def forward(self, logits: np.ndarray):

        logits = logits.tolist()

        B = len(logits)
        N = len(logits[0])
        K = self.config.k
        temp = max(self.config.temperature, 1e-6)

        hard = []
        indices = []

        for b in range(B):

            scaled = [x / temp for x in logits[b]]
            taken = [0.0] * N

            hard_b = []
            idx_b = []

            for _ in range(K):

                best_i = max(
                    range(N),
                    key=lambda i: scaled[i] + np.log(1.0 - taken[i] + 1e-8)
                )

                vec = [0.0] * N
                vec[best_i] = 1.0

                hard_b.append(vec)
                idx_b.append(best_i)

                taken[best_i] = 1.0

            hard.append(hard_b)
            indices.append(idx_b)

        return np.array(hard, dtype=np.float32), np.array(indices, dtype=np.int32)


class TopKNumpySTE(TopKBackend):
    """
    Hard top-k in forward, softmax relaxation in backward.
    """

    def forward(self, logits: np.ndarray) -> TopKResult:
        x = np.ascontiguousarray(logits, dtype=np.float32)

        B, N = x.shape
        K = self.config.k
        temp = max(self.config.temperature, 1e-6)

        scaled = x / temp

        idx_part = np.argpartition(scaled, -K, axis=1)[:, -K:]
        top_vals = np.take_along_axis(scaled, idx_part, axis=1)
        order = np.argsort(-top_vals, axis=1)
        indices = np.take_along_axis(idx_part, order, axis=1).astype(np.int32)

        hard = np.zeros((B, N), dtype=np.float32)
        row_ids = np.arange(B)[:, None]
        hard[row_ids, indices] = 1.0

        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp = np.exp(shifted)
        soft = exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)
        soft = soft.astype(np.float32, copy=False)

        return TopKResult(
            hard=hard,
            soft=soft,
            indices=indices,
            scores=x,
        )

    def backward(self, grad_out: np.ndarray, cache: TopKResult) -> np.ndarray:
        """
        Backward through the soft relaxation only.
        grad_out: dL/dy where y is treated as soft selection weights (B, N)
        returns dL/dlogits (B, N)
        """
        grad_out = np.ascontiguousarray(grad_out, dtype=np.float32)
        soft = cache.soft
        temp = max(self.config.temperature, 1e-6)

        dot = np.sum(grad_out * soft, axis=1, keepdims=True)
        grad_logits = (soft * (grad_out - dot)) / temp
        return np.ascontiguousarray(grad_logits, dtype=np.float32)


try:
    from src.parallel.cython_topk import greedy_topk_fast
    CYTHON_AVAILABLE = True
except Exception:
    CYTHON_AVAILABLE = False


class TopKCythonSTE(TopKBackend):
    def __init__(self, config: TopKConfig):
        super().__init__(config)
        if not CYTHON_AVAILABLE:
            raise RuntimeError("Cython backend not available")

    def forward(self, logits: np.ndarray) -> TopKResult:
        x = np.ascontiguousarray(logits, dtype=np.float32)
        B, N = x.shape
        K = self.config.k
        temp = max(self.config.temperature, 1e-6)

        scaled = np.ascontiguousarray(x / temp, dtype=np.float32)

        indices = greedy_topk_fast(scaled, K)
        indices = np.ascontiguousarray(indices, dtype=np.int32)

        hard = np.zeros((B, N), dtype=np.float32)
        rows = np.arange(B)[:, None]
        hard[rows, indices] = 1.0

        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp = np.exp(shifted)
        soft = exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)
        soft = np.ascontiguousarray(soft, dtype=np.float32)

        return TopKResult(
            hard=hard,
            soft=soft,
            indices=indices,
            scores=x,
        )

    def backward(self, grad_out: np.ndarray, cache: TopKResult) -> np.ndarray:
        grad_out = np.ascontiguousarray(grad_out, dtype=np.float32)
        soft = cache.soft
        temp = max(self.config.temperature, 1e-6)

        dot = np.sum(grad_out * soft, axis=1, keepdims=True)
        grad_logits = (soft * (grad_out - dot)) / temp
        return np.ascontiguousarray(grad_logits, dtype=np.float32)


def build_topk(config: TopKConfig, backend: str = "numpy") -> TopKBackend:

    if backend == "numpy":
        return TopKNumpySTE(config)

    if backend == "python":
        return TopKNaive(config)

    if backend == "cython":
        return TopKCythonSTE(config)

    raise ValueError(f"Unknown backend: {backend}")


def main():

    logging.basicConfig(level=logging.INFO)

    B, N = 64, 2048
    logits = np.random.randn(B, N).astype(np.float32)

    cfg = TopKConfig(k=8, temperature=1.0)

    logger.info("Benchmark NumPy")
    m2 = TopKNumpySTE(cfg)
    t2 = m2.benchmark(logits)

    logger.info("Benchmark Naive")
    m1 = TopKNaive(cfg)
    t1 = m1.benchmark(logits)

    logger.info(f"NumPy: {t2:.6f}s | Naive: {t1:.6f}s")

    if CYTHON_AVAILABLE:
        logger.info("Benchmark Cython")
        m3 = TopKCythonSTE(cfg)
        t3 = m3.benchmark(logits)

        logger.info(f"Cython: {t3:.6f}s")
        logger.info(f"Speedup NumPy→Cython: {t2 / t3:.2f}x")


if __name__ == "__main__":
    main()