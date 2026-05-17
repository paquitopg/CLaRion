import time
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math

sys.path.insert(0, os.getcwd())
logger = logging.getLogger("clara_topk")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


CYTHON_AVAILABLE = False

try:
    from src.parallel.cython_topk import greedy_topk_fast
    CYTHON_AVAILABLE = True
    logger.info("Cython backend loaded (greedy_topk_fast)")
except Exception as e:
    CYTHON_AVAILABLE = False
    logger.exception(f"Cython not available: {e}")


class ClaraTopKNaive:
    def __init__(self, k: int, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature

    def forward(self, logits: torch.Tensor):

        logits = logits.detach().cpu().tolist()
        B = len(logits)
        N = len(logits[0])
        K = self.k

        hard = []
        indices = []

        for b in range(B):

            scaled = [x / max(self.temperature, 1e-6) for x in logits[b]]

            taken = [0.0] * N
            hard_b = []
            idx_b = []

            for _ in range(K):

                # argmax
                best_i = max(range(N), key=lambda i: scaled[i] + math.log(1.0 - taken[i] + 1e-8))

                hard_vec = [0.0] * N
                hard_vec[best_i] = 1.0

                hard_b.append(hard_vec)
                idx_b.append(best_i)

                taken[best_i] = 1.0

            hard.append(hard_b)
            indices.append(idx_b)

        hard = torch.tensor(hard)
        indices = torch.tensor(indices)

        return hard, indices


class ClaraTopKNumpy:
    def __init__(self, k: int, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature

    def forward(self, logits: torch.Tensor):

        x = logits.detach().cpu().numpy()
        B, N = x.shape
        K = self.k

        scaled = x / max(self.temperature, 1e-6)

        hard = np.zeros((B, K, N), dtype=np.float32)
        indices = np.zeros((B, K), dtype=np.int32)

        taken = np.zeros((B, N), dtype=np.float32)

        for b in range(B):
            for j in range(K):

                masked = scaled[b] + np.log(1.0 - taken[b] + 1e-8)

                idx = int(np.argmax(masked))

                hard[b, j, idx] = 1.0
                indices[b, j] = idx
                taken[b, idx] = 1.0

        return torch.from_numpy(hard), torch.from_numpy(indices)


class ClaraTopKCython(nn.Module):
    def __init__(self, k: int):
        super().__init__()

        if not CYTHON_AVAILABLE:
            raise RuntimeError("Cython extension not available")

        self.k = k

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        logits_np = logits.detach().contiguous().cpu().numpy()

        indices_np = greedy_topk_fast(logits_np, self.k)

        indices = torch.from_numpy(indices_np).to(logits.device)

        B, N = logits.shape

        hard = torch.zeros(B, self.k, N, device=logits.device, dtype=logits.dtype)
        hard.scatter_(2, indices.unsqueeze(-1), 1.0)

        soft = torch.zeros_like(hard)
        taken = torch.zeros_like(logits)

        for j in range(self.k):
            mask = 1.0 - taken.detach()
            masked = logits + torch.log(mask + 1e-8)
            soft[:, j] = F.softmax(masked, dim=-1)
            taken = torch.clamp(taken + hard[:, j], max=1.0)

        return hard + (soft - soft.detach()), indices


def benchmark(model: nn.Module, logits: torch.Tensor, iters: int = 50) -> float:

    with torch.no_grad():
        for _ in range(5):
            model(logits)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iters):
            model(logits)

    end = time.perf_counter()

    return (end - start) / iters


def main():

    torch.manual_seed(0)

    B, N, K = 64, 2048, 8
    logits = torch.randn(B, N)

    logger.info(f"Benchmark setup: B={B}, N={N}, K={K}")

    logger.info("Running NumPy version")
    m2 = ClaraTopKNumpy(K)
    t2 = benchmark(m2, logits)
    logger.info(f"NumPy time: {t2:.6f}s")

    logger.info("Running Naive Python version")
    m1 = ClaraTopKNaive(K)
    t1 = benchmark(m1, logits)
    logger.info(f"Naive time: {t1:.6f}s")

    if CYTHON_AVAILABLE:
        logger.info("Running Cython version")
        m3 = ClaraTopKCython(K)
        t3 = benchmark(m3, logits)

        logger.info(f"Cython time: {t3:.6f}s")

        logger.info(f"Speedup Numpy → Cython: {t2 / t3:.2f}x")
        logger.info(f"Speedup Naive → Cython: {t1 / t3:.2f}x")

    else:
        logger.info("Cython skipped")


if __name__ == "__main__":
    main()