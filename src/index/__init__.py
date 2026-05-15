"""
CLaRiON document index.

Offline pipeline: documents -> tokens -> encoder -> compressed vectors -> bank.
The bank is frozen and shared with the retriever at query time, matching the
CLaRa design where doc embeddings are pre-computed once and never re-encoded.
"""

from .builder import IndexBuilder, encode_corpus_serial, encode_corpus_parallel
from .store import IndexStore
from .scorer import (
    cosine_python_loop,
    cosine_numpy,
    cosine_cython_omp,
    top_k_indices,
    Retriever,
)

__all__ = [
    "IndexBuilder",
    "IndexStore",
    "encode_corpus_serial",
    "encode_corpus_parallel",
    "cosine_python_loop",
    "cosine_numpy",
    "cosine_cython_omp",
    "top_k_indices",
    "Retriever",
]
