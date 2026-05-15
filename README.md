# CLaRiON

**C**ontinuous **L**atent **A**ugmented-**R**etrieval **I**nference **O**n **N**-cores.

A CPU-parallel reimplementation of [CLaRa](https://arxiv.org/abs/2511.18659) at toy scale. The point is to take CLaRa's three compute hotspots — document encoding, cosine retrieval over a corpus, and the differentiable Straight-Through
top-k — and parallelize each on CPU with Cython + OpenMP, then benchmark
against a naive baseline.

Authors: Avner El Baz Paco Goze.

---

## What it does

Pipeline mirrors CLaRa, end-to-end:

1. **Encoder** — a 2-layer transformer compresses each document into a small
   set of memory-token embeddings (`src/models/encoder.py`);
2. **Index** — embeddings stored as a frozen `(N, D)` bank
   (`src/index/store.py`);
3. **Retriever** — cosine similarity over the bank + top-k selection
   (`src/index/scorer.py`);
4. **Differentiable top-k** — Straight-Through estimator that lets the
   generator's loss propagate back to the retriever (`src/models/topk.py`);
5. **Decoder** — generator that conditions on the retrieved memory tokens.

Each compute hotspot has three implementations side by side for the
benchmark report: a pure-Python loop, a numpy/BLAS baseline, and a
Cython + OpenMP parallel version.

---

## Quick start

Requirements: Python 3.14, `uv`, and (on macOS) `libomp` from Homebrew.

```bash
# 1. Install OpenMP (macOS only — Linux clang/gcc ship it).
brew install libomp

# 2. Sync dependencies and build the Cython + OpenMP extensions.
uv sync
uv run python setup.py build_ext --inplace

# 3. Verify all three extensions import.
uv run python -c "from src.parallel import cython_encoder, cython_index, cython_topk; print('ok')"
```

## Run the benchmarks

Each script writes timings to `reports/*.json` and prints a speedup table.

```bash
# Encoder forward pass, numpy vs Cython at 1/2/4/8 threads.
uv run python -m src.benchmarks.bench_encoder

# Cosine retrieval over corpus, pure-Python vs numpy vs Cython.
uv run python -m src.benchmarks.bench_index

# Full pipeline: offline corpus encoding + online retrieval.
uv run python -m src.benchmarks.bench_pipeline

# Sanity check: numpy <-> Cython agree, and outputs plug into ClaraTopK.
uv run python -m src.benchmarks.integration_check
```

## Run on a real corpus

```bash
# Pull 10k Wikipedia paragraphs (HuggingFace `datasets`).
uv run python -m src.data.cli pull-wiki --n 10000 --out data/wiki_10k.txt

# Re-run the encoder benchmark on real text.
uv run python -m src.data.cli bench-on-real \
    --source file --file-path data/wiki_10k.txt --n-docs 5000
```

`datasets` is an optional dependency; if it's not installed, all loaders
fall back to a small bundled sample of 20 Wikipedia-flavored paragraphs.

---

## Layout

```
src/
  models/
    config.py        ModelConfig / IndexConfig / BenchConfig
    memory.py        Learnable memory-token bank
    encoder.py       Tiny 2-layer transformer (numpy + Cython backends)
    topk.py          Differentiable Straight-Through top-k (Avner)
    ...              decoder, attention, generation, lora, ... (Avner, WIP)
  index/
    builder.py       Offline doc encoding (serial + multiprocessing)
    store.py         On-disk embedding bank
    scorer.py        Cosine + top-k retrieval, three backends
  parallel/
    cython_encoder.pyx   OpenMP-parallel encoder hot kernels
    cython_index.pyx     OpenMP-parallel cosine + top-k
    cython_topk.pyx      OpenMP-parallel quickselect top-k (Avner)
  data/
    loaders.py       HuggingFace + bundled corpus / QA loaders
    cli.py           pull-wiki / pull-qa / bench-on-real
  benchmarks/        numpy-vs-Cython timing harnesses
  pipeline.py        End-to-end glue (encoder -> retrieval -> ClaraTopK)

setup.py             Builds all three Cython extensions in-place.
```

---

## Implementation notes

- **Parallelism axis.** OpenMP `prange` over the batch axis for the encoder
  and over the corpus axis for the retriever.
- **Same weights across backends.** `build_encoder(...)` accepts a shared
  `params=`, so the numpy baseline and the Cython version produce
  bit-comparable outputs (~1e-6 max diff, float32 epsilon).
- **No training.** Per the course brief, correctness doesn't matter; only
  speed does. Encoder weights are random-initialized and never updated.
- **Cython 3 reduction trap.** All `+=` patterns inside `prange` bodies are
  written `acc = acc + ...` to avoid Cython's OpenMP-reduction inference,
  which would otherwise forbid reading the accumulator in the same loop
  iteration.

---

## References

- Paper: Jie He et al., *CLaRa: Bridging Retrieval and Generation with
  Continuous Latent Reasoning*, arXiv:2511.18659 (Feb 2026).
- Apple's reference repo: <https://github.com/apple/ml-clara>.
