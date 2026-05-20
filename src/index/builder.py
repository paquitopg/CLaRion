from __future__ import annotations

import json
import logging
import os
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


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
    return enc["input_ids"].astype(np.int32, copy=False)


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
    t0 = time.perf_counter()
    print(f"[worker:init] backend={backend} threads={threads}", flush=True)
    _GLOBAL_ENCODER = build_encoder(
        config,
        backend=backend,
        num_threads=threads,
        params=params,
    )
    dt = time.perf_counter() - t0
    print(f"[worker:init] encoder ready elapsed={dt:.4f}s", flush=True)


def _encode_worker(batch):
    t0 = time.perf_counter()
    print(f"[worker:encode] batch_shape={batch.shape}", flush=True)
    out = _GLOBAL_ENCODER.forward(batch)
    dt = time.perf_counter() - t0
    print(f"[worker:encode] out_shape={out.shape} elapsed={dt:.4f}s", flush=True)
    return out


def encode_corpus_parallel(token_ids, config, params, backend, batch_size):
    n_chunks = max(1, (len(token_ids) + batch_size - 1) // batch_size)
    n_workers = min(n_chunks, 4, max(1, cpu_count() - 1))

    chunks = [
        np.ascontiguousarray(token_ids[i:i + batch_size], dtype=np.int32)
        for i in range(0, len(token_ids), batch_size)
    ]

    print(
        f"[encode_corpus_parallel] backend={backend} "
        f"n_workers={n_workers} n_chunks={len(chunks)} "
        f"batch_size={batch_size}",
        flush=True,
    )

    if len(chunks) < 4:
        print("[encode_corpus_parallel] fallback=serial_small_workload", flush=True)
        encoder = build_encoder(
            config,
            backend=backend,
            num_threads=1,
            params=params,
        )
        return encode_corpus_serial(token_ids, encoder, batch_size)

    t0 = time.perf_counter()
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(config, params, backend, 1),
    ) as pool:
        results = []
        for idx, out in enumerate(pool.imap(_encode_worker, chunks), start=1):
            print(
                f"[encode_corpus_parallel] chunk={idx}/{len(chunks)} done "
                f"shape={out.shape}",
                flush=True,
            )
            results.append(out)

    bank = np.ascontiguousarray(np.concatenate(results, axis=0), dtype=np.float32)
    dt = time.perf_counter() - t0
    print(f"[encode_corpus_parallel] total_elapsed={dt:.4f}s", flush=True)
    return bank


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
        t_tok0 = time.perf_counter()
        token_ids = tokenize_corpus(
            docs,
            self.tokenizer,
            cfg.max_seq_len,
        )
        t_tok1 = time.perf_counter()
        print(
            f"[IndexBuilder.build] tokenize:done shape={token_ids.shape} "
            f"elapsed={t_tok1 - t_tok0:.4f}s",
            flush=True,
        )

        t0 = time.perf_counter()

        print(f"[IndexBuilder.build] build_encoder:start backend={backend}", flush=True)
        t_enc0 = time.perf_counter()
        encoder = build_encoder(
            cfg,
            backend=backend,
            params=self.params,
            num_threads=1 if backend == "cython" else 0,
        )
        t_enc1 = time.perf_counter()
        print(
            f"[IndexBuilder.build] build_encoder:done type={type(encoder).__name__} "
            f"elapsed={t_enc1 - t_enc0:.4f}s",
            flush=True,
        )

        self.params = encoder.params

        use_parallel = bool(parallel and backend == "cython")
        print(f"[IndexBuilder.build] encode:start parallel={use_parallel}", flush=True)

        t_bank0 = time.perf_counter()
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
        t_bank1 = time.perf_counter()

        print(
            f"[IndexBuilder.build] encode:done bank_shape={bank.shape} "
            f"elapsed={t_bank1 - t_bank0:.4f}s",
            flush=True,
        )

        wall = time.perf_counter() - t0

        t_ret0 = time.perf_counter()
        retrieval_bank = flatten_memory_bank(bank)
        retrieval_bank = np.ascontiguousarray(
            l2_normalize(retrieval_bank),
            dtype=np.float32,
        )
        t_ret1 = time.perf_counter()
        print(
            f"[IndexBuilder.build] retrieval_bank:done shape={retrieval_bank.shape} "
            f"elapsed={t_ret1 - t_ret0:.4f}s",
            flush=True,
        )

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