"""
End-to-end pipeline benchmark: tokenize -> encode corpus -> retrieve top-k.

Mirrors the offline-then-online split of CLaRa:

    Phase A (offline, one-shot, large): tokenize corpus + encode with the
       compressor.  Compared across:
         - numpy + serial
         - numpy + multiprocessing
         - cython+OpenMP + serial
         - cython+OpenMP + multiprocessing

    Phase B (online, per-query):  cosine over the bank + top-k.
       Compared across numpy vs cython.

This is the script Xavier's report will be built around: it ties the
encoder-side parallelization story to the index-side one and shows how the
two compose.

Run:

    PYTHONPATH=. python -m src.benchmarks.bench_pipeline
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src.index.builder import (
    encode_corpus_parallel,
    encode_corpus_serial,
    make_synthetic_corpus,
    tokenize_corpus,
)
from src.index.scorer import (
    cosine_cython_omp,
    cosine_numpy,
    top_k_indices,
)
from src.models.config import ModelConfig
from src.models.encoder import build_encoder

from ._timing import TimingResult, print_table, save_results, time_call

logging.getLogger("clarion").setLevel(logging.WARNING)


def run(args) -> None:
    cfg = ModelConfig(
        hidden_dim=args.hidden, n_layers=args.layers, n_heads=args.heads,
        ffn_dim=args.ffn, n_memory_tokens=args.memory_tokens,
        max_doc_len=args.doc_len, vocab_size=8_000,
    )

    print(f"\n== Phase A: offline corpus encoding ==  (N={args.n_docs}, "
          f"{cfg.n_layers}L, h={cfg.hidden_dim}, doc_len={cfg.max_doc_len})")
    docs = make_synthetic_corpus(args.n_docs, words_per_doc=cfg.max_doc_len)
    token_ids = tokenize_corpus(docs, cfg)

    # Build a single set of params and reuse across backends.
    base = build_encoder(cfg, backend="numpy")
    params = base.params

    rows: list[TimingResult] = []

    # 1) numpy + serial
    rows.append(time_call(
        lambda: encode_corpus_serial(token_ids, base, batch_size=args.batch),
        warmup=1, iters=args.iters,
        label="encode numpy serial",
        metadata={"backend": "numpy", "parallel": False},
    ))

    # 2) numpy + multiprocessing
    rows.append(time_call(
        lambda: encode_corpus_parallel(
            token_ids, cfg, params=params, backend="numpy",
            n_workers=args.workers, chunk_size=args.batch,
        ),
        warmup=1, iters=args.iters,
        label=f"encode numpy + mp(w={args.workers})",
        metadata={"backend": "numpy", "parallel": True, "workers": args.workers},
    ))

    # 3) cython + serial @ N threads
    for nt in args.thread_counts:
        cy = build_encoder(cfg, backend="cython", num_threads=nt, params=params)
        if not cy.available:
            print("Cython backend missing — skip cython benches")
            break
        rows.append(time_call(
            lambda enc=cy: encode_corpus_serial(token_ids, enc, batch_size=args.batch),
            warmup=1, iters=args.iters,
            label=f"encode cython(nt={nt}) serial",
            metadata={"backend": "cython", "threads": nt, "parallel": False},
        ))

    # 4) cython + multiprocessing (each worker uses 1 OpenMP thread to avoid
    #    over-subscription; this is the recommended composition).
    rows.append(time_call(
        lambda: encode_corpus_parallel(
            token_ids, cfg, params=params, backend="cython",
            n_workers=args.workers, chunk_size=args.batch, inner_threads=1,
        ),
        warmup=1, iters=args.iters,
        label=f"encode cython(1) + mp(w={args.workers})",
        metadata={"backend": "cython", "threads": 1, "parallel": True,
                  "workers": args.workers},
    ))

    print_table(rows, baseline_label=None)

    # --------------------------------------------------------------------- #
    # Phase B: online retrieval.
    # --------------------------------------------------------------------- #
    print(f"\n== Phase B: online retrieval ==  (corpus_dim={cfg.embedding_dim}, "
          f"Q={args.n_queries})")

    bank = encode_corpus_serial(token_ids, base, batch_size=args.batch)
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(args.n_queries, cfg.embedding_dim)).astype(np.float32)

    rows_b: list[TimingResult] = []
    rows_b.append(time_call(
        lambda: top_k_indices(cosine_numpy(queries, bank), k=args.k, backend="numpy"),
        warmup=2, iters=args.iters,
        label="retrieve numpy", metadata={"backend": "numpy"},
    ))
    for nt in args.thread_counts:
        rows_b.append(time_call(
            lambda t=nt: top_k_indices(
                cosine_cython_omp(queries, bank, t), k=args.k,
                backend="cython", num_threads=t,
            ),
            warmup=2, iters=args.iters,
            label=f"retrieve cython(nt={nt})",
            metadata={"backend": "cython", "threads": nt},
        ))

    print_table(rows_b, baseline_label=None)

    # Save both phases.
    all_rows = (
        [TimingResult(**{**r.__dict__, "label": "A | " + r.label}) for r in rows]
        + [TimingResult(**{**r.__dict__, "label": "B | " + r.label}) for r in rows_b]
    )
    save_results(all_rows, args.out)
    print(f"\nSaved JSON to {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-docs", type=int, default=2_000)
    p.add_argument("--n-queries", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--thread-counts", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--doc-len", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn", type=int, default=128)
    p.add_argument("--memory-tokens", type=int, default=4)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--out", type=str, default="reports/pipeline_bench.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
