"""
Shared timing utilities for the encoder + index benchmarks.

We deliberately keep this 50 lines of pure stdlib — `timeit` is a fine API
but its output is not what the report needs. We want median + p95 + speedup
across multiple thread counts in a single table.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable


@dataclass
class TimingResult:
    label: str
    median_s: float
    p95_s: float
    min_s: float
    samples: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def median_ms(self) -> float:
        return self.median_s * 1000.0


def time_call(fn: Callable[[], object], warmup: int = 3, iters: int = 10,
              label: str = "", metadata: dict | None = None) -> TimingResult:
    """Warm up `warmup` times, then time `iters` runs.

    Wall-clock only — `perf_counter` is the right tool for ms-scale CPU work.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return TimingResult(
        label=label,
        median_s=statistics.median(samples),
        p95_s=samples[max(0, int(0.95 * len(samples)) - 1)],
        min_s=min(samples),
        samples=samples,
        metadata=metadata or {},
    )


def print_table(rows: list[TimingResult], baseline_label: str | None = None) -> None:
    """Pretty-print a comparison table to stdout."""
    if not rows:
        print("(no rows)")
        return
    baseline = next((r for r in rows if r.label == baseline_label), rows[0])
    header = f"{'backend':<28} {'median (ms)':>12} {'p95 (ms)':>10} {'min (ms)':>10} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        speedup = baseline.median_s / r.median_s if r.median_s > 0 else float("inf")
        print(f"{r.label:<28} {r.median_ms:>12.3f} {r.p95_s*1000:>10.3f} "
              f"{r.min_s*1000:>10.3f} {speedup:>8.2f}x")


def save_results(rows: list[TimingResult], path: str) -> None:
    """Dump results to JSON for later plotting."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
