"""
Benchmark: encoder forward pass, numpy vs Cython+OpenMP.

Sweep:
  * batch size in {16, 64, 256}
  * thread count in {1, 2, 4, 8} (Cython backend only)

Outputs a console table and a JSON dump in reports/encoder_bench.json.

Run from the project root:

    PYTHONPATH=. python -m src.benchmarks.bench_encoder
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src.models.config import ModelConfig
from src.models.encoder import build_encoder

from ._timing import TimingResult, print_table, save_results, time_call


logging.getLogger("clarion").setLevel(logging.WARNING)


def make_inputs(cfg: ModelConfig, batch: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, cfg.vocab_size, size=(batch, cfg.max_doc_len), dtype=np.int32)


def run(args) -> None:
    cfg = ModelConfig(
        vocab_size=8_000,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
        ffn_dim=args.ffn,
        n_memory_tokens=args.memory_tokens,
        max_doc_len=args.doc_len,
    )

    # Pre-build numpy encoder so we can share params with cython.
    numpy_enc = build_encoder(cfg, backend="numpy")
    params = numpy_enc.params

    rows: list[TimingResult] = []

    for batch in args.batch_sizes:
        ids = make_inputs(cfg, batch)

        # ---- numpy baseline ----
        r = time_call(
            lambda enc=numpy_enc, x=ids: enc.forward(x),
            warmup=args.warmup, iters=args.iters,
            label=f"numpy             B={batch:<4}",
            metadata={"batch": batch, "backend": "numpy"},
        )
        rows.append(r)

        # ---- cython @ N threads ----
        for nt in args.thread_counts:
            cy_enc = build_encoder(cfg, backend="cython", num_threads=nt, params=params)
            if not cy_enc.available:
                print("Cython encoder not available — skipping.")
                break
            r = time_call(
                lambda enc=cy_enc, x=ids: enc.forward(x),
                warmup=args.warmup, iters=args.iters,
                label=f"cython(nt={nt})       B={batch:<4}",
                metadata={"batch": batch, "backend": "cython", "threads": nt},
            )
            rows.append(r)

    print()
    print(f"== Encoder forward benchmark ==  ({cfg.n_layers}-layer, hidden={cfg.hidden_dim}, "
          f"heads={cfg.n_heads}, ffn={cfg.ffn_dim}, mem={cfg.n_memory_tokens}, doc_len={cfg.max_doc_len})")
    print_table(rows, baseline_label=None)

    save_results(rows, args.out)
    print(f"\nSaved JSON to {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 64, 256])
    p.add_argument("--thread-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--doc-len", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn", type=int, default=512)
    p.add_argument("--memory-tokens", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--out", type=str, default="reports/encoder_bench.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
