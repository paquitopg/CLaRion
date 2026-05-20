from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RetrievalCache:
    query: np.ndarray
    retrieval_bank: np.ndarray
    structured_bank: np.ndarray
    topk_cache: Any
    memory: np.ndarray


class ClaraPipeline:
    """
    End-to-end retrieval + generation pipeline.

    Hybrid design:
    - encoder: produces query embeddings
    - topk: differentiable selection backend (STE-capable)
    - decoder: consumes retrieved memory
    """

    def __init__(self, encoder: Any, decoder: Any, topk: Any):
        self.encoder = encoder
        self.decoder = decoder
        self.topk = topk
        self._last_cache: RetrievalCache | None = None

    def _normalize_bank(self, bank: np.ndarray) -> np.ndarray:
        """
        Normalize retrieval bank to unit vectors.

        Args:
            bank: (N, T, H)

        Returns:
            retrieval_bank: (N, H)
        """
        bank = np.ascontiguousarray(bank, dtype=np.float32)

        x = bank.mean(axis=1)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / (norms + 1e-12)

        return np.ascontiguousarray(x, dtype=np.float32)

    def _retrieve(
        self,
        query: np.ndarray,
        retrieval_bank: np.ndarray,
        structured_bank: np.ndarray,
    ) -> tuple[np.ndarray, Any]:
        """
        Retrieve top-k memory blocks.

        Args:
            query: (B, H)
            retrieval_bank: (N, H)
            structured_bank: (N, T, H)

        Returns:
            memory: (B, k*T, H)
            topk_result: TopKResult
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        retrieval_bank = np.ascontiguousarray(retrieval_bank, dtype=np.float32)
        structured_bank = np.ascontiguousarray(structured_bank, dtype=np.float32)

        scores = np.ascontiguousarray(query @ retrieval_bank.T, dtype=np.float32)
        topk_result = self.topk.forward(scores)

        indices = np.ascontiguousarray(topk_result.indices, dtype=np.int32)
        B, k = indices.shape
        _, T, H = structured_bank.shape

        memory = structured_bank[indices]
        memory = np.ascontiguousarray(memory.reshape(B, k * T, H), dtype=np.float32)

        return memory, topk_result

    def forward(
        self,
        input_ids: np.ndarray,
        bank: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            input_ids: (B, T_in)
            bank: (N, T_mem, H)

        Returns:
            logits: decoder output
            indices: retrieved doc ids
        """
        input_ids = np.ascontiguousarray(input_ids, dtype=np.int32)
        bank = np.ascontiguousarray(bank, dtype=np.float32)

        retrieval_bank = self._normalize_bank(bank)
        query = np.ascontiguousarray(self.encoder.encode_retrieval(input_ids), dtype=np.float32)

        memory, topk_result = self._retrieve(
            query=query,
            retrieval_bank=retrieval_bank,
            structured_bank=bank,
        )

        logits = self.decoder.forward(input_ids, memory)

        self._last_cache = RetrievalCache(
            query=query,
            retrieval_bank=retrieval_bank,
            structured_bank=bank,
            topk_cache=topk_result,
            memory=memory,
        )

        return logits, topk_result.indices

    def backward(
        self,
        input_ids: np.ndarray,
        grad_memory: np.ndarray,
        lr: float = 1e-3,
    ) -> np.ndarray:
        """
        Backward path from decoder memory gradient to encoder query head.

        Args:
            input_ids: (B, T_in)
            grad_memory: dL/dmemory, shape (B, k*T, H)
            lr: learning rate for encoder.backward_query

        Returns:
            grad_query: dL/dquery, shape (B, H)
        """
        if self._last_cache is None:
            raise RuntimeError("Call forward() before backward().")

        input_ids = np.ascontiguousarray(input_ids, dtype=np.int32)
        grad_memory = np.ascontiguousarray(grad_memory, dtype=np.float32)

        cache = self._last_cache
        topk_cache = cache.topk_cache
        retrieval_bank = np.ascontiguousarray(cache.retrieval_bank, dtype=np.float32)

        if grad_memory.ndim != 3:
            raise ValueError(f"grad_memory must be 3D (B, k*T, H), got shape={grad_memory.shape}")

        if topk_cache.indices.ndim != 2:
            raise ValueError(
                f"topk_cache.indices must be 2D (B, k), got shape={topk_cache.indices.shape}"
            )

        B, kT, H = grad_memory.shape
        B_topk, k = topk_cache.indices.shape

        if B != B_topk:
            raise ValueError(
                f"Batch mismatch between grad_memory ({B}) and topk indices ({B_topk})"
            )

        if k <= 0:
            raise ValueError(f"top-k must be > 0, got k={k}")

        if kT % k != 0:
            raise ValueError(
                f"grad_memory second dim must be divisible by k: got kT={kT}, k={k}"
            )

        T = kT // k

        grad_selected = grad_memory.reshape(B, k, T, H)

        grad_selected_docs = grad_selected.mean(axis=2)
        grad_selected_scores = np.linalg.norm(grad_selected_docs, axis=-1)
        grad_selected_scores = grad_selected_scores.astype(np.float32, copy=False)

        grad_topk_out = np.zeros_like(topk_cache.hard, dtype=np.float32)
        rows = np.arange(B)[:, None]
        grad_topk_out[rows, topk_cache.indices] = grad_selected_scores

        grad_scores = self.topk.backward(grad_topk_out, topk_cache)
        grad_scores = np.ascontiguousarray(grad_scores, dtype=np.float32)

        grad_query = np.ascontiguousarray(grad_scores @ retrieval_bank, dtype=np.float32)

        if not np.isfinite(grad_query).all():
            raise FloatingPointError("grad_query contains NaN or Inf.")

        self.encoder.backward_query(input_ids, grad_query, lr=lr)

        return grad_query

    def generate(
        self,
        input_ids: np.ndarray,
        bank: np.ndarray,
        **gen_kwargs,
    ) -> np.ndarray:
        """
        Autoregressive generation with retrieval memory.
        """
        input_ids = np.ascontiguousarray(input_ids, dtype=np.int32)
        bank = np.ascontiguousarray(bank, dtype=np.float32)

        retrieval_bank = self._normalize_bank(bank)
        query = np.ascontiguousarray(self.encoder.encode_retrieval(input_ids), dtype=np.float32)

        memory, _ = self._retrieve(
            query=query,
            retrieval_bank=retrieval_bank,
            structured_bank=bank,
        )

        return self.decoder.generate(input_ids, memory, **gen_kwargs)