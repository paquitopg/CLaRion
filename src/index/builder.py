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
from .scorer import flatten_memory_bank, l2_normalize
from .store import IndexStore

logger = logging.getLogger("clarion.index.builder")


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


def _iter_batches(x: np.ndarray, batch_size: int):
    for i in range(0, x.shape[0], batch_size):
        yield i, min(i + batch_size, x.shape[0]), x[i:i + batch_size]


def encode_corpus_serial(token_ids, encoder, batch_size: int):
    cfg = encoder.config
    out = np.empty(
        (token_ids.shape[0], cfg.n_memory_tokens, cfg.hidden_dim),
        dtype=np.float32,
    )

    for i, j, batch in _iter_batches(token_ids, batch_size):
        out[i:j] = encoder.forward(batch)

    return np.ascontiguousarray(out, dtype=np.float32)


_GLOBAL_ENCODER = None


def _init_worker(config, params, backend, threads):
    global _GLOBAL_ENCODER
    print(f"[worker:init] backend={backend} threads={threads}", flush=True)
    _GLOBAL_ENCODER = build_encoder(
        config,
        backend=backend,
        num_threads=threads,
        params=params,
    )
    print("[worker:init] encoder ready", flush=True)


def _encode_worker(batch):
    print(f"[worker:encode] batch_shape={batch.shape}", flush=True)
    out = _GLOBAL_ENCODER.forward(batch)
    print(f"[worker:encode] out_shape={out.shape}", flush=True)
    return out


def encode_corpus_parallel(token_ids, config, params, backend, batch_size):
    n_workers = min(4, max(1, cpu_count() - 1))

    chunks = [
        np.ascontiguousarray(token_ids[i:i + batch_size], dtype=np.int32)
        for i in range(0, len(token_ids), batch_size)
    ]

    print(
        f"[encode_corpus_parallel] backend={backend} "
        f"n_workers={n_workers} n_chunks={len(chunks)}",
        flush=True,
    )

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(config, params, backend, 1),
        maxtasksperchild=1,
    ) as pool:
        results = []
        for idx, out in enumerate(pool.imap(_encode_worker, chunks), start=1):
            print(
                f"[encode_corpus_parallel] chunk={idx}/{len(chunks)} done "
                f"shape={out.shape}",
                flush=True,
            )
            results.append(out)

    return np.ascontiguousarray(np.concatenate(results, axis=0), dtype=np.float32)


@dataclass
class BuildReport:
    n_docs: int
    n_memory_tokens: int
    hidden_dim: int
    embedding_dim: int
    backend: str
    wall_time_s: float
    docs_per_s: float


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

        print(f"[IndexBuilder.build] tokenize:start n_docs={len(docs)}", flush=True)
        token_ids = tokenize_corpus(
            docs,
            self.tokenizer,
            cfg.max_seq_len,
        )
        print(f"[IndexBuilder.build] tokenize:done shape={token_ids.shape}", flush=True)

        t0 = time.perf_counter()

        print(f"[IndexBuilder.build] build_encoder:start backend={backend}", flush=True)
        encoder = build_encoder(
            cfg,
            backend=backend,
            params=self.params,
        )
        print(f"[IndexBuilder.build] build_encoder:done type={type(encoder).__name__}", flush=True)

        self.params = encoder.params

        use_parallel = bool(parallel and backend == "cython")
        print(f"[IndexBuilder.build] encode:start parallel={use_parallel}", flush=True)

        if use_parallel:
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

        print(f"[IndexBuilder.build] encode:done bank_shape={bank.shape}", flush=True)

        wall = time.perf_counter() - t0

        retrieval_bank = flatten_memory_bank(bank)
        retrieval_bank = np.ascontiguousarray(l2_normalize(retrieval_bank), dtype=np.float32)

        report = BuildReport(
            n_docs=len(docs),
            n_memory_tokens=cfg.n_memory_tokens,
            hidden_dim=cfg.hidden_dim,
            embedding_dim=retrieval_bank.shape[1],
            backend=backend,
            wall_time_s=wall,
            docs_per_s=len(docs) / wall if wall > 0 else float("inf"),
        )

        if save:
            store = IndexStore(
                self.index_config.index_path,
                self.index_config.meta_path,
            )

            store.save(
                retrieval_bank,
                cfg,
                extra={
                    "build_report": report.__dict__,
                    "memory_bank_shape": list(bank.shape),
                },
            )

            np.save(
                Path(self.index_config.index_path).with_name("memory_bank.npy"),
                bank,
            )

            np.save(
                Path(self.index_config.index_path).with_name("retrieval_bank.npy"),
                retrieval_bank,
            )

            np.save(
                Path(self.index_config.index_path).with_name("token_ids.npy"),
                token_ids,
            )

            with Path(self.index_config.index_path).with_name("raw_docs.jsonl").open("w", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps({"text": d}) + "\n")

        return bank, report