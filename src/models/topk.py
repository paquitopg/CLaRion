import time
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

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



class ClaraTopK(nn.Module):
    def __init__(self, k: int, temperature: float = 1.0):
        super().__init__()
        self.k = k
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N = logits.shape
        scaled = logits / max(self.temperature, 1e-6)

        _, topk_idx = torch.topk(scaled, self.k, dim=-1)

        hard = torch.zeros(B, self.k, N, device=logits.device, dtype=logits.dtype)
        hard.scatter_(2, topk_idx.unsqueeze(-1), 1.0)

        soft = torch.empty_like(hard)
        taken = torch.zeros_like(logits)

        for j in range(self.k):
            mask = 1.0 - taken.detach()
            masked = scaled + torch.log(mask + 1e-8)
            soft[:, j] = F.softmax(masked, dim=-1)
            taken = torch.clamp(taken + hard[:, j], max=1.0)

        return hard + (soft - soft.detach()), topk_idx


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

    logger.info("Running PyTorch baseline")
    m1 = ClaraTopK(K)

    out1, idx1 = m1(logits)
    t1 = benchmark(m1, logits)

    logger.info(f"PyTorch time: {t1:.6f}s")

    if CYTHON_AVAILABLE:

        logger.info("Running Cython optimized version")
        m2 = ClaraTopKCython(K)

        out2, idx2 = m2(logits)
        t2 = benchmark(m2, logits)

        logger.info(f"Cython time: {t2:.6f}s")
        logger.info(f"Speedup: {t1 / t2:.2f}x")

    else:
        logger.info("Cython benchmark skipped")


if __name__ == "__main__":
    main()