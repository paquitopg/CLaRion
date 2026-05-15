"""
Benchmark: cosine similarity over the corpus, three backends.

Sweep:
  * corpus size in {1k, 10k, 50k}
  * thread count in {1, 2, 4, 8} (Cython backend only)

The pure-Python backend is only run for the smallest corpus (it scales as
O(Q*N*D) in interpreted code and would take minutes at N=10k).

Run:

    PYTHONPATH=. python -m src.benchmarks.bench_index
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src.index.scorer import (
    cosine_cython_omp,
    cosine_numpy,
    cosine_python_loop,
    top_k_indices,
)

from ._timing import TimingResult, print_table, save_results, time_call

logging.getLogger("clarion").setLevel(logging.WARNING)


def make_index(n_docs: int, dim: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    corpus = rng.normal(size=(n_docs, dim)).astype(np.float32)
    queries = rng.normal(size=(32, dim)).astype(np.float32)
    return queries, corpus


def run(args) -> None:
    rows: list[TimingResult] = []

    for n in args.corpus_sizes:
        queries, corpus = make_index(n, args.dim)

        # ---- numpy baseline ----
        r = time_call(
            lambda q=queries, m=corpus: cosine_numpy(q, m),
            warmup=args.warmup, iters=args.iters,
            label=f"numpy             N={n:<6}",
            metadata={"N": n, "backend": "numpy"},
        )
        rows.append(r)

        # ---- pure-python only for tiny corpora ----
        if n <= args.python_n_cap:
            r = time_call(
                lambda q=queries[:4], m=corpus: cosine_python_loop(q, m),
                warmup=1, iters=2,
                label=f"python-loop (Q=4) N={n:<6}",
                metadata={"N": n, "backend": "python", "Q": 4},
            )
            rows.append(r)

        # ---- cython @ N threads ----
        for nt in args.thread_counts:
            r = time_call(
                lambda q=queries, m=corpus, t=nt: cosine_cython_omp(q, m, t),
                warmup=args.warmup, iters=args.iters,
                label=f"cython(nt={nt})       N={n:<6}",
                metadata={"N": n, "backend": "cython", "threads": nt},
            )
            rows.append(r)

        # ---- end-to-end retrieval (scores + top-k) ----
        def search_numpy(q=queries, m=corpus):
            return top_k_indices(cosine_numpy(q, m), k=args.k, backend="numpy")

        def search_cython(q=queries, m=corpus, t=args.thread_counts[-1]):
            return top_k_indices(
                cosine_cython_omp(q, m, t), k=args.k, backend="cython", num_threads=t,
            )

        rows.append(time_call(
            search_numpy, warmup=args.warmup, iters=args.iters,
            label=f"e2e numpy         N={n:<6}",
            metadata={"N": n, "backend": "numpy+numpy-topk", "k": args.k},
        ))
        rows.append(time_call(
            search_cython, warmup=args.warmup, iters=args.iters,
            label=f"e2e cython        N={n:<6}",
            metadata={"N": n, "backend": "cython+cython-topk",
                      "k": args.k, "threads": args.thread_counts[-1]},
        ))

    print()
    print(f"== Index cosine retrieval benchmark ==  (Q=32 queries, dim={args.dim}, k={args.k})")
    print_table(rows, baseline_label=None)
    save_results(rows, args.out)
    print(f"\nSaved JSON to {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-sizes", nargs="+", type=int, default=[1_000, 10_000, 50_000])
    p.add_argument("--thread-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--dim", type=int, default=1024)  # = n_memory_tokens * hidden_dim
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--python-n-cap", type=int, default=1_000,
                   help="Max corpus size at which to run the pure-Python loop.")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", type=str, default="reports/index_bench.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
