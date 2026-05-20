from __future__ import annotations

from typing import Any, Tuple
import numpy as np


class ClaraPipeline:
    """
    End-to-end retrieval + generation pipeline.

    Fully backend-driven:
    - encoder: produces query embeddings (NumPy)
    - decoder: consumes retrieved memory
    - topk: shared selection backend (NumPy/Cython/Python)
    """

    def __init__(self, encoder: Any, decoder: Any, topk: Any):
        self.encoder = encoder
        self.decoder = decoder
        self.topk = topk

    def _normalize_bank(self, bank: np.ndarray) -> np.ndarray:
        """
        Normalize retrieval bank to unit vectors.

        Args:
            bank: (N, T, H)

        Returns:
            (N, H) normalized embeddings
        """
        x = bank.mean(axis=1).astype(np.float32)
        x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        return x

    def _retrieve(
        self,
        query: np.ndarray,
        retrieval_bank: np.ndarray,
        structured_bank: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k memory blocks.

        Args:
            query: (B, H)
            retrieval_bank: (N, H)
            structured_bank: (N, T, H)

        Returns:
            memory: (B, k*T, H)
            indices: (B, k)
        """

        scores = query @ retrieval_bank.T

        _, indices = self.topk.forward(scores)

        B, k = indices.shape
        _, T, H = structured_bank.shape

        memory = structured_bank[indices]
        memory = memory.reshape(B, k * T, H)

        return memory, indices

    def forward(
        self,
        input_ids: np.ndarray,
        bank: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            input_ids: (B, T)
            bank: (N, T, H)

        Returns:
            logits: decoder output
            indices: retrieved doc ids
        """

        retrieval_bank = self._normalize_bank(bank)

        query = self.encoder.encode_retrieval(input_ids)

        memory, indices = self._retrieve(
            query=query,
            retrieval_bank=retrieval_bank,
            structured_bank=bank,
        )

        logits = self.decoder.forward(input_ids, memory)

        return logits, indices

    def generate(
        self,
        input_ids: np.ndarray,
        bank: np.ndarray,
        **gen_kwargs,
    ) -> np.ndarray:
        """
        Autoregressive generation with retrieval memory.
        """

        retrieval_bank = self._normalize_bank(bank)

        query = self.encoder.encode_retrieval(input_ids)

        memory, _ = self._retrieve(
            query=query,
            retrieval_bank=retrieval_bank,
            structured_bank=bank,
        )

        return self.decoder.generate(input_ids, memory, **gen_kwargs)