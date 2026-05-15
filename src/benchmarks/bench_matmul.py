"""
Matmul micro-benchmark: naive Cython vs cache-blocked Cython vs cache-blocked
+ #pragma omp simd Cython vs BLAS-backed numpy.

Why this benchmark exists
-------------------------
The encoder forward pass spends most of its CPU time inside the FFN matmul
(W1 @ x and W2 @ x), so the matmul kernel is the right thing to optimize.
This script measures three CPU-parallel matmul variants in isolation, prints
a GFLOPS table for the report, and emits a JSON dump for later plotting.

The three variants we expose:

  1. naive     —  outer i-k-j triple loop, `prange` over rows.  No tiling.
  2. blocked   —  BM×BK×BN cache tiling (see tile_sizes() in cython_encoder).
                  Inner micro-kernel left to the C compiler's auto-vectorizer.
  3. +SIMD     —  Same tiling, but the micro-kernel's innermost loop is
                  annotated with `#pragma omp simd`, forcing explicit
                  NEON/AVX vector loads/FMAs even when the compiler's cost
                  model would skip vectorization.

Numpy `@` is included as the ceiling — BLAS (Accelerate on macOS, OpenBLAS or
MKL on Linux) is the practical upper bound for dense float32 matmul on CPU.
Our goal is to understand the gap, not close it.

Run:

    PYTHONPATH=. python -m src.benchmarks.bench_matmul
    PYTHONPATH=. python -m src.benchmarks.bench_matmul --include-large
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from ._timing import TimingResult, print_table, save_results, time_call

try:
    from src.parallel import cython_encoder  # type: ignore
    CY_AVAILABLE = True
except Exception as e:  # pragma: no cover
    cython_encoder = None
    CY_AVAILABLE = False
    print(f"[fatal] cython_encoder extension not built: {e}", file=sys.stderr)
    print("Run `uv run python setup.py build_ext --inplace` first.", file=sys.stderr)
    sys.exit(1)


logging.getLogger("clarion").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Shape catalog
# --------------------------------------------------------------------------- #
# (M, K, N, label) — sized to mirror what the encoder actually feeds the FFN.
# B * L  ×  H  ×  F   for FFN-up,  B * L  ×  F  ×  H   for FFN-down.
DEFAULT_SHAPES = [
    # encoder-realistic
    ( 256,  64, 128, "B=16,L=16, FFN-up    H=64,F=128"),
    ( 256, 128,  64, "B=16,L=16, FFN-down  F=128,H=64"),
    (1024, 128, 512, "B=64,L=16, FFN-up    H=128,F=512"),
    (1024, 512, 128, "B=64,L=16, FFN-down  F=512,H=128"),
    # larger — where tiling theoretically wins
    (2048, 1024, 1024, "synthetic 2K×1K×1K"),
]

LARGE_K_SHAPES = [
    # K dominates — pushes a row of A out of L1 to force tile reuse.
    (256,  8192, 256, "K-dominated 256×8K×256"),
    (256, 16384, 256, "K-dominated 256×16K×256"),
    (512, 16384, 512, "K-dominated 512×16K×512"),
]


# --------------------------------------------------------------------------- #
# Bench
# --------------------------------------------------------------------------- #
def run(args) -> None:
    BM, BK, BN = cython_encoder.tile_sizes()
    print(f"Tile sizes: BM={BM}, BK={BK}, BN={BN}\n")

    shapes = list(DEFAULT_SHAPES)
    if args.include_large:
        shapes.extend(LARGE_K_SHAPES)

    rng = np.random.default_rng(0)
    rows: list[TimingResult] = []

    for M, K, N, label in shapes:
        A = rng.normal(size=(M, K)).astype(np.float32)
        B = rng.normal(size=(K, N)).astype(np.float32)
        flops = 2.0 * M * K * N

        print(f"== {label}  ({M}×{K}×{N}) ==")
        sub_rows: list[TimingResult] = []

        # BLAS ceiling
        r = time_call(
            lambda A=A, B=B: A @ B,
            warmup=args.warmup, iters=args.iters,
            label="numpy/BLAS",
            metadata={"M": M, "K": K, "N": N, "backend": "blas", "flops": flops},
        )
        sub_rows.append(r)

        for nt in args.thread_counts:
            r = time_call(
                lambda A=A, B=B, t=nt: cython_encoder.matmul_naive(A, B, t),
                warmup=args.warmup, iters=args.iters,
                label=f"naive       nt={nt}",
                metadata={"M": M, "K": K, "N": N, "backend": "cython_naive",
                          "threads": nt, "flops": flops},
            )
            sub_rows.append(r)
            r = time_call(
                lambda A=A, B=B, t=nt: cython_encoder.matmul_blocked(A, B, t),
                warmup=args.warmup, iters=args.iters,
                label=f"blocked     nt={nt}",
                metadata={"M": M, "K": K, "N": N, "backend": "cython_blocked",
                          "threads": nt, "flops": flops},
            )
            sub_rows.append(r)
            r = time_call(
                lambda A=A, B=B, t=nt: cython_encoder.matmul_blocked_simd(A, B, t),
                warmup=args.warmup, iters=args.iters,
                label=f"+SIMD       nt={nt}",
                metadata={"M": M, "K": K, "N": N, "backend": "cython_blocked_simd",
                          "threads": nt, "flops": flops},
            )
            sub_rows.append(r)

        # Pretty-print this shape's table, with GFLOPS column appended.
        _print_with_gflops(sub_rows)
        rows.extend(sub_rows)
        print()

    save_results(rows, args.out)
    print(f"Saved JSON to {args.out}")


def _print_with_gflops(rows: list[TimingResult]) -> None:
    """Per-shape table with GFLOPS so it's the headline metric."""
    if not rows:
        return
    baseline = rows[0]  # numpy/BLAS is row 0 in our layout
    header = f"{'backend':<22} {'median (ms)':>12} {'GFLOPS':>10} {'vs BLAS':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        flops = float(r.metadata.get("flops", 0.0))
        gflops = flops / (r.median_s * 1e9) if r.median_s > 0 else 0.0
        ratio = baseline.median_s / r.median_s if r.median_s > 0 else float("inf")
        print(f"{r.label:<22} {r.median_ms:>12.3f} {gflops:>10.1f} {ratio:>8.2f}x")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--thread-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--include-large",
                   action="store_true",
                   help="Also run K-dominated shapes where tiling should win.")
    p.add_argument("--out", type=str, default="reports/matmul_bench.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
