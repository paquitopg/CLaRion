from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from transformers import AutoTokenizer

from ..models.config import IndexConfig, ModelConfig
from ..models.encoder import EncoderParams, build_encoder
from .scorer import l2_normalize
from .store import IndexStore

logger = logging.getLogger("clarion.index.builder")


# =========================================================
# TOKENIZER (SHARED GLOBAL CONTRACT)
# =========================================================
def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    return tok


def tokenize_corpus(docs, tokenizer, max_len: int):
    enc = tokenizer(
        list(docs),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int32)


# =========================================================
# HELPERS
# =========================================================
def flatten_memory_bank(bank: np.ndarray) -> np.ndarray:
    assert bank.ndim == 3
    pooled = bank.mean(axis=1) 

    return np.ascontiguousarray(
        l2_normalize(pooled),
        dtype=np.float32
    )

def _iter_batches(x: np.ndarray, batch_size: int):
    for i in range(0, x.shape[0], batch_size):
        yield i, min(i + batch_size, x.shape[0]), x[i:i + batch_size]


# =========================================================
# ENCODING (SINGLE PROCESS)
# =========================================================
def encode_corpus_serial(token_ids, encoder, batch_size: int):
    cfg = encoder.config

    out = np.empty(
        (token_ids.shape[0], cfg.n_memory_tokens, cfg.hidden_dim),
        dtype=np.float32,
    )

    for i, j, batch in _iter_batches(token_ids, batch_size):
        out[i:j] = encoder.forward(batch)

    return np.ascontiguousarray(out, dtype=np.float32)


# =========================================================
# MULTIPROCESS
# =========================================================
_GLOBAL_ENCODER = None


def _init_worker(config, params, backend, threads):
    global _GLOBAL_ENCODER
    _GLOBAL_ENCODER = build_encoder(
        config,
        backend=backend,
        num_threads=threads,
        params=params,
    )


def _encode_worker(batch):
    return _GLOBAL_ENCODER.forward(batch)


def encode_corpus_parallel(token_ids, config, params, backend, batch_size):
    n_workers = max(1, cpu_count() - 1)

    chunks = [
        token_ids[i:i + batch_size]
        for i in range(0, len(token_ids), batch_size)
    ]

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(config, params, backend, 1),
    ) as pool:
        results = pool.map(_encode_worker, chunks)

    return np.concatenate(results, axis=0)


# =========================================================
# REPORT
# =========================================================
@dataclass
class BuildReport:
    n_docs: int
    n_memory_tokens: int
    hidden_dim: int
    embedding_dim: int
    backend: str
    wall_time_s: float
    docs_per_s: float


# =========================================================
# INDEX BUILDER
# =========================================================
class IndexBuilder:
    def __init__(
        self,
        model_config: ModelConfig,
        index_config: IndexConfig,
        tokenizer_name: str,
        params: Optional[EncoderParams] = None,
    ):
        self.model_config = model_config
        self.index_config = index_config
        self.params = params

        self.tokenizer = build_tokenizer(tokenizer_name)

    def build(
        self,
        docs: Sequence[str],
        backend: str = "numpy",
        parallel: bool = False,
        save: bool = True,
    ):

        cfg = self.model_config

        # ---------------------------
        # TOKENIZATION
        # ---------------------------
        token_ids = tokenize_corpus(
            docs,
            self.tokenizer,
            cfg.max_seq_len,
        )

        # ---------------------------
        # ENCODING
        # ---------------------------
        t0 = time.perf_counter()

        encoder = build_encoder(
            cfg,
            backend=backend,
            params=self.params,
        )

        self.params = encoder.params

        if parallel:
            bank = encode_corpus_parallel(
                token_ids,
                cfg,
                self.params,
                backend,
                self.index_config.batch_size,
            )
        else:
            bank = encode_corpus_serial(
                token_ids,
                encoder,
                self.index_config.batch_size,
            )

        wall = time.perf_counter() - t0

        # ---------------------------
        # RETRIEVAL BANK
        # ---------------------------
        retrieval_bank = flatten_memory_bank(bank)

        # ---------------------------
        # REPORT
        # ---------------------------
        report = BuildReport(
            n_docs=len(docs),
            n_memory_tokens=cfg.n_memory_tokens,
            hidden_dim=cfg.hidden_dim,
            embedding_dim=cfg.embedding_dim,
            backend=backend,
            wall_time_s=wall,
            docs_per_s=len(docs) / wall if wall > 0 else float("inf"),
        )

        # ---------------------------
        # SAVE
        # ---------------------------
        if save:
            store = IndexStore(
                self.index_config.index_path,
                self.index_config.meta_path,
            )

            store.save(
                bank,
                cfg,
                extra={"build_report": report.__dict__},
            )

            np.save(
                Path(self.index_config.index_path).with_name("retrieval_bank.npy"),
                retrieval_bank,
            )

            np.save(
                Path(self.index_config.index_path).with_name("token_ids.npy"),
                token_ids,
            )

            with Path(self.index_config.index_path).with_name("raw_docs.jsonl").open("w") as f:
                for d in docs:
                    f.write(json.dumps({"text": d}) + "\n")

        return bank, report