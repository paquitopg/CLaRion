# CLaRiON — Encoder + Index side (Paco)

This half of CLaRiON owns everything from raw documents to a frozen embedding
bank plus a fast cosine-similarity retriever. The decoder side (Avner) owns
the differentiable top-k aggregator (`src/parallel/cython_topk.pyx`,
`src/models/topk.py`) and the generator.

## Build

From the project root, once per environment:

```bash
pip install -U cython numpy setuptools
python setup.py build_ext --inplace
```

This compiles two OpenMP-parallel Cython extensions in place:

- `src/parallel/cython_encoder` — encoder hot kernels (matmul, attention, FFN)
- `src/parallel/cython_index`   — cosine + top-k retrieval kernels

Avner's `cython_topk` is built separately by `src/parallel/setup.py` and is not
touched by this build.

## Module map

```
src/models/
  config.py        ModelConfig + IndexConfig + BenchConfig dataclasses
  memory.py        Learnable memory-token bank (l × hidden)
  encoder.py       Tiny 2-layer transformer compressor.
                   Two backends: EncoderNumpy, EncoderCython.

src/index/
  builder.py       Offline doc-encoding pipeline.
                   Backends: serial, multiprocessing.
                   Includes a deterministic synthetic-corpus generator.
  store.py         IndexStore: on-disk doc embedding bank (.npy + meta json).
  scorer.py        Cosine + top-k. 3 backends: python-loop / numpy / cython.

src/parallel/
  cython_encoder.pyx   OpenMP encoder block forward (prange over batch).
  cython_index.pyx     OpenMP cosine + top-k (prange over corpus / queries).

src/benchmarks/
  bench_encoder.py   numpy vs cython, sweep over batch & thread count.
  bench_index.py     pure-Python vs numpy vs cython cosine, sweep over N.
  bench_pipeline.py  end-to-end (Phase A offline encoding + Phase B online
                     retrieval). The script that powers the report's plots.
```

## Run benchmarks

```bash
PYTHONPATH=. python -m src.benchmarks.bench_encoder
PYTHONPATH=. python -m src.benchmarks.bench_index
PYTHONPATH=. python -m src.benchmarks.bench_pipeline
```

Each saves a JSON file under `reports/` for later plotting.

## Design choices worth knowing

- **Parallelism axis.** OpenMP `prange` is on the batch axis for the encoder
  and on the corpus axis for the retriever. Both are embarrassingly parallel,
  which gives the cleanest speedup story in the report.
- **Per-thread scratch.** The encoder block kernel allocates a single
  `(B × per_doc_scratch)` slab once per call, and each thread slices its own
  region indexed by the batch loop variable. No locks, no shared writes.
- **Same weights across backends.** `build_encoder(...)` accepts a shared
  `params=` so the numpy reference and the Cython version produce
  bit-comparable outputs (modulo float32 epsilon — confirmed ~1e-6 max diff).
- **Reduction-variable trap.** All `+=` patterns inside `prange` bodies were
  rewritten as `acc = acc + ...` to avoid Cython 3's reduction inference.
  See the comment block in `cython_encoder.pyx::_rms_norm_rows`.
- **No training.** Per Xavier's email ("correctness doesn't matter, just
  speed"), encoder weights are random-init and never updated.

## Interface for the decoder side (Avner)

The contract Avner can rely on:

```python
from src.models.config import ModelConfig
from src.models.encoder import build_encoder
from src.index.scorer import Retriever

cfg = ModelConfig()
encoder = build_encoder(cfg, backend="cython", num_threads=4)

# Phase A (offline, once):
bank = encoder.forward(tokenized_corpus)          # (N, l*H) float32
retriever = Retriever(bank)

# Phase B (online, per query):
result = retriever.search(query_embeddings,       # (Q, l*H) float32
                          k=5,
                          backend="cython",
                          num_threads=4)
# result.indices : (Q, k) int32
# result.scores  : (Q, k) float32
```

The `indices` returned from `Retriever.search` are exactly what
`ClaraTopK` / `ClaraTopKCython` should consume on the decoder side.
