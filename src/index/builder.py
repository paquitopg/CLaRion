"""
Index builder: offline corpus -> embedding bank.

This is the most parallel-friendly part of the whole CLaRiON pipeline because
every document is independent — the canonical "embarrassingly parallel" case.
We expose three flavors so the benchmark report has a clean speedup curve:

  * `encode_corpus_serial`     : single-process, single-thread baseline.
  * `encode_corpus_parallel`   : multiprocessing.Pool, one worker per chunk.
  * `IndexBuilder.build_cython`: serial process, but the per-batch encoder
                                 uses OpenMP threads inside Cython.

The cleanest comparison for the report is then:
    (a) pure-numpy serial
    (b) numpy + multiprocessing
    (c) Cython+OpenMP
    (d) Cython+OpenMP + multiprocessing  (composite, if cores are abundant)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Iterable, Optional, Sequence

import numpy as np

from ..models.config import IndexConfig, ModelConfig
from ..models.encoder import EncoderBackend, EncoderParams, build_encoder
from .store import IndexStore

logger = logging.getLogger("clarion.index.builder")


# --------------------------------------------------------------------------- #
# Toy tokenizer
# --------------------------------------------------------------------------- #
def hash_tokenize(text: str, max_len: int, vocab_size: int, pad_id: int) -> np.ndarray:
    """
    Stable hash-tokenizer for our compute-speed study.

    The professor's brief explicitly says answer quality does not matter, so we
    use a deterministic word-hash that needs no vocabulary file. Every word
    maps to a slot in [1, vocab_size). Slot 0 is the pad id by convention.
    """
    tokens: list[int] = []
    for word in text.split():
        # Cheap stable hash — Python's hash() varies between runs (PYTHONHASHSEED).
        h = 0
        for ch in word.lower():
            h = (h * 1315423911) ^ ord(ch)
            h &= 0xFFFFFFFF
        # Reserve 0 for pad.
        tokens.append((h % (vocab_size - 1)) + 1)
        if len(tokens) >= max_len:
            break
    if len(tokens) < max_len:
        tokens.extend([pad_id] * (max_len - len(tokens)))
    return np.asarray(tokens, dtype=np.int32)


def tokenize_corpus(docs: Sequence[str], config: ModelConfig) -> np.ndarray:
    """Tokenize a whole corpus into a (N, L) int32 matrix."""
    out = np.empty((len(docs), config.max_doc_len), dtype=np.int32)
    for i, d in enumerate(docs):
        out[i] = hash_tokenize(d, config.max_doc_len, config.vocab_size, config.pad_id)
    return out


# --------------------------------------------------------------------------- #
# Encoding loops
# --------------------------------------------------------------------------- #
def _iter_batches(token_ids: np.ndarray, batch_size: int) -> Iterable[tuple[int, int, np.ndarray]]:
    n = token_ids.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end, token_ids[start:end]


def encode_corpus_serial(
    token_ids: np.ndarray,
    encoder: EncoderBackend,
    batch_size: int = 64,
) -> np.ndarray:
    """Single-process, single-thread encoding loop. The reference baseline."""
    cfg = encoder.config
    out = np.empty((token_ids.shape[0], cfg.embedding_dim), dtype=np.float32)
    for start, end, batch in _iter_batches(token_ids, batch_size):
        out[start:end] = encoder.forward(batch)
    return out


# Workers used by encode_corpus_parallel. Module-level so they pickle.
_GLOBAL_WORKER_ENCODER: Optional[EncoderBackend] = None
_GLOBAL_WORKER_CONFIG: Optional[ModelConfig] = None


def _worker_init(
    config: ModelConfig,
    params: Optional[EncoderParams],
    backend: str,
    num_threads: int,
) -> None:
    """Build one encoder per worker process (called once on Pool init)."""
    global _GLOBAL_WORKER_ENCODER, _GLOBAL_WORKER_CONFIG
    _GLOBAL_WORKER_CONFIG = config
    _GLOBAL_WORKER_ENCODER = build_encoder(
        config, backend=backend, num_threads=num_threads, params=params,
    )


def _worker_encode(chunk_token_ids: np.ndarray) -> np.ndarray:
    assert _GLOBAL_WORKER_ENCODER is not None, "Pool worker was not initialized"
    return _GLOBAL_WORKER_ENCODER.forward(chunk_token_ids)


def encode_corpus_parallel(
    token_ids: np.ndarray,
    config: ModelConfig,
    params: Optional[EncoderParams] = None,
    backend: str = "numpy",
    n_workers: Optional[int] = None,
    chunk_size: int = 128,
    inner_threads: int = 1,
) -> np.ndarray:
    """
    Multiprocessing encode. Each worker holds its own encoder instance.

    Why multiprocessing not threading? Numpy is mostly GIL-free for BLAS but
    the pure-Python control flow around it (slicing, dtype casts, attribute
    access) isn't. multiprocessing gets us linear CPU scaling for the
    embarrassingly-parallel case at the cost of a model copy per worker — fine
    for our toy 2-layer model.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    chunks: list[np.ndarray] = []
    for _, _, batch in _iter_batches(token_ids, chunk_size):
        chunks.append(batch)

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(config, params, backend, inner_threads),
    ) as pool:
        results = pool.map(_worker_encode, chunks, chunksize=1)

    return np.concatenate(results, axis=0)


# --------------------------------------------------------------------------- #
# High-level builder
# --------------------------------------------------------------------------- #
@dataclass
class BuildReport:
    """Returned by IndexBuilder.build — useful for the bench report."""
    n_docs: int
    embedding_dim: int
    backend: str
    wall_time_s: float
    docs_per_s: float


class IndexBuilder:
    """End-to-end "documents -> stored bank" runner."""

    def __init__(
        self,
        model_config: ModelConfig,
        index_config: IndexConfig,
        params: Optional[EncoderParams] = None,
    ):
        self.model_config = model_config
        self.index_config = index_config
        self.params = params  # shared across backends for apples-to-apples

    def build(
        self,
        docs: Sequence[str],
        backend: str = "numpy",
        parallel: bool = False,
        n_workers: Optional[int] = None,
        inner_threads: int = 0,
        save: bool = True,
    ) -> tuple[np.ndarray, BuildReport]:
        cfg = self.model_config
        logger.info(
            "Building index: docs=%d backend=%s parallel=%s n_workers=%s threads=%d",
            len(docs), backend, parallel, n_workers, inner_threads,
        )

        token_ids = tokenize_corpus(docs, cfg)
        t0 = time.perf_counter()
        if parallel:
            bank = encode_corpus_parallel(
                token_ids, cfg, params=self.params, backend=backend,
                n_workers=n_workers, chunk_size=self.index_config.batch_size,
                inner_threads=inner_threads,
            )
        else:
            encoder = build_encoder(
                cfg, backend=backend, num_threads=inner_threads, params=self.params,
            )
            # Make sure subsequent calls in the same process keep the same params.
            self.params = encoder.params
            bank = encode_corpus_serial(token_ids, encoder, batch_size=self.index_config.batch_size)
        wall = time.perf_counter() - t0

        report = BuildReport(
            n_docs=bank.shape[0],
            embedding_dim=bank.shape[1],
            backend=backend,
            wall_time_s=wall,
            docs_per_s=bank.shape[0] / wall if wall > 0 else float("inf"),
        )
        logger.info(
            "Index built: %d docs in %.2fs (%.1f docs/s)",
            report.n_docs, report.wall_time_s, report.docs_per_s,
        )

        if save:
            store = IndexStore(self.index_config.index_path, self.index_config.meta_path)
            store.save(bank, cfg, extra={"build_report": report.__dict__})

        return bank, report


# --------------------------------------------------------------------------- #
# Quick synthetic-data helper used by benchmarks
# --------------------------------------------------------------------------- #
def make_synthetic_corpus(n_docs: int, words_per_doc: int = 64, seed: int = 0) -> list[str]:
    """
    Generate a deterministic synthetic corpus of `n_docs` documents.

    Uses a Zipf-ish word generator over a 5k vocab so the resulting documents
    look "wikipedia-shaped" to the tokenizer (high-frequency function words,
    long tail of rare words). Used in the benchmark scripts only.
    """
    rng = np.random.default_rng(seed)
    vocab = [f"w{i:05d}" for i in range(5000)]
    # Zipf-like sampling weights
    weights = 1.0 / np.arange(1, len(vocab) + 1)
    weights /= weights.sum()
    out: list[str] = []
    for _ in range(n_docs):
        ids = rng.choice(len(vocab), size=words_per_doc, p=weights)
        out.append(" ".join(vocab[i] for i in ids))
    return out
