"""
CLaRiON data ingestion.

Optional module — the parallelization benchmarks don't strictly need it,
but using real Wikipedia paragraphs (vs the synthetic Zipf-shaped generator
in `index/builder.py`) makes the report numbers more credible and unlocks
Avner's training stub.

Loaders pull from HuggingFace `datasets` when it's available, and fall back
to a small bundled sample of real Wikipedia paragraphs (committed in
`fallback_sample.py`) when the dependency is missing or the network is
offline. This keeps the benchmark smoke tests hermetic.
"""

from .loaders import (
    load_corpus,
    load_qa_dataset,
    DocumentCorpus,
    QAExample,
    HF_AVAILABLE,
)

__all__ = [
    "load_corpus",
    "load_qa_dataset",
    "DocumentCorpus",
    "QAExample",
    "HF_AVAILABLE",
]
