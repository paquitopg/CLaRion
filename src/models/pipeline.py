from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .topk import ClaraTopKCython


def search_differentiable(
    retrieval_bank: np.ndarray,
    structured_bank: np.ndarray,
    query_embeddings: torch.Tensor,
    topk: ClaraTopKCython,
):
    """
    Compute top-k retrieval over a cosine similarity space and gather
    corresponding memory representations.

    Args:
        retrieval_bank: (B, H) normalized document embeddings.
        structured_bank: (B, T, H) full memory-token representations.
        query_embeddings: (B, H) query embeddings.
        topk: differentiable top-k operator.

    Returns:
        selected: (B, k, H) retrieved memory representations.
        indices: (B, k) selected document indices.
    """

    device = query_embeddings.device

    retrieval_bank_t = torch.from_numpy(retrieval_bank).to(device)
    q = F.normalize(query_embeddings, dim=-1)

    scores = q @ retrieval_bank_t.T
    Z, indices = topk(scores)

    memory = structured_bank.mean(axis=1)
    memory = torch.from_numpy(memory).to(device)

    selected = Z @ memory

    return selected, indices


class ClaraPipeline:
    """
    Retrieval-augmented generation pipeline.

    Combines:
        - encoder producing query embeddings
        - cosine retrieval over a memory bank
        - differentiable top-k selection
        - decoder conditioned on retrieved memory
    """

    def __init__(self, encoder, decoder, topk):
        self.encoder = encoder
        self.decoder = decoder
        self.topk = topk

    def forward(self, input_ids: torch.Tensor, bank: np.ndarray):
        """
        Forward pass through retrieval-augmented decoder.

        Args:
            input_ids: (B, L) token ids.
            bank: (N, T, H) encoded document memory bank.

        Returns:
            logits: (B, L, V)
            indices: (B, k) retrieved document indices
        """

        retrieval_bank = bank.mean(axis=1)
        retrieval_bank = retrieval_bank.astype(np.float32, copy=False)
        retrieval_bank = np.ascontiguousarray(retrieval_bank)

        norm = np.linalg.norm(retrieval_bank, axis=-1, keepdims=True) + 1e-12
        retrieval_bank = retrieval_bank / norm

        query_mem = self.encoder.encode_retrieval(
            input_ids.cpu().numpy()
        )

        query_mem = torch.from_numpy(query_mem).to(input_ids.device)

        memory, idx = search_differentiable(
            retrieval_bank=retrieval_bank,
            structured_bank=bank,
            query_embeddings=query_mem,
            topk=self.topk,
        )

        logits = self.decoder(
            input_ids=input_ids,
            memory=memory,
        )

        return logits, idx