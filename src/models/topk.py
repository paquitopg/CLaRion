from __future__ import annotations

import time
import logging
from typing import Tuple, Optional

import numpy as np

from .config import TopKConfig

logger = logging.getLogger("clara_topk")


class TopKBackend:
    """
    Abstract Top-K backend (NumPy/Cython only).
    """

    def __init__(self, config: TopKConfig):
        self.config = config

    def forward(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


class TopKNumpy(TopKBackend):
    """
    Vectorized NumPy greedy Top-K.
    """

    def forward(self, logits: np.ndarray):

        x = logits

        B, N = x.shape
        K = self.config.k
        temp = max(self.config.temperature, 1e-6)

        scaled = x / temp

        hard = np.zeros((B, K, N), dtype=np.float32)
        indices = np.zeros((B, K), dtype=np.int32)
        taken = np.zeros((B, N), dtype=np.float32)

        for b in range(B):
            for j in range(K):

                scores = scaled[b] + np.log(1.0 - taken[b] + 1e-8)
                idx = int(np.argmax(scores))

                hard[b, j, idx] = 1.0
                indices[b, j] = idx
                taken[b, idx] = 1.0

        return hard, indices


try:
    from src.parallel.cython_topk import greedy_topk_fast
    CYTHON_AVAILABLE = True
except Exception:
    CYTHON_AVAILABLE = False


class TopKCython(TopKBackend):
    """
    Cython accelerated greedy Top-K.
    """

    def __init__(self, config: TopKConfig):
        super().__init__(config)

        if not CYTHON_AVAILABLE:
            raise RuntimeError("Cython backend not available")

    def forward(self, logits: np.ndarray):

        indices = greedy_topk_fast(logits, self.config.k)

        B, N = logits.shape
        K = self.config.k

        hard = np.zeros((B, K, N), dtype=np.float32)
        taken = np.zeros((B, N), dtype=np.float32)

        for b in range(B):
            for j in range(K):
                idx = indices[b, j]
                hard[b, j, idx] = 1.0
                taken[b, idx] = 1.0

        return hard, indices


def build_topk(config: TopKConfig, backend: str = "numpy") -> TopKBackend:

    if backend == "numpy":
        return TopKNumpy(config)

    if backend == "python":
        return TopKNaive(config)

    if backend == "cython":
        return TopKCython(config)

    raise ValueError(f"Unknown backend: {backend}")


def main():

    logging.basicConfig(level=logging.INFO)

    B, N = 64, 2048
    logits = np.random.randn(B, N).astype(np.float32)

    cfg = TopKConfig(k=8, temperature=1.0)

    logger.info("Benchmark NumPy")
    m2 = TopKNumpy(cfg)
    t2 = m2.benchmark(logits)

    logger.info("Benchmark Naive")
    m1 = TopKNaive(cfg)
    t1 = m1.benchmark(logits)

    logger.info(f"NumPy: {t2:.6f}s | Naive: {t1:.6f}s")

    if CYTHON_AVAILABLE:
        logger.info("Benchmark Cython")
        m3 = TopKCython(cfg)
        t3 = m3.benchmark(logits)

        logger.info(f"Cython: {t3:.6f}s")
        logger.info(f"Speedup NumPy→Cython: {t2 / t3:.2f}x")


if __name__ == "__main__":
    main()