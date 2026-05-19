"""
Data loaders for CLaRiON.

Three sources for the corpus:

  * "hf"        — HuggingFace `datasets`, the same path Apple's CLaRa uses
                  for Wikipedia-2021 and the QA benchmarks. Requires the
                  `datasets` library (optional dep).
  * "bundled"   — the ~20-paragraph sample committed at
                  `fallback_sample.BUNDLED_DOCS`. Hermetic, no network.
  * "file"      — a local plain-text file, one paragraph per line. Useful
                  for caching an HF slice to avoid re-downloading.

Three QA datasets supported (all via HF):

  * Natural Questions (`google-research-datasets/nq_open`)
  * HotpotQA           (`hotpotqa/hotpot_qa`, distractor variant)
  * MuSiQue            (`musique`)
  * 2WikiMultiHopQA    (`2wiki_multihop_qa`)

All loaders are streaming-friendly so that the encoder/index benchmark
can keep its memory footprint flat as N grows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

logger = logging.getLogger("clarion.data.loaders")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

try:
    from datasets import load_dataset  # type: ignore
    HF_AVAILABLE = True
except Exception as e:  # pragma: no cover
    load_dataset = None  # type: ignore
    HF_AVAILABLE = False
    logger.info("HuggingFace `datasets` not available (%s); HF loaders disabled.", e)


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class DocumentCorpus:
    """A passive container for a list of plain-text documents."""
    docs: list[str]
    source: str
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.docs)

    def __iter__(self) -> Iterator[str]:
        return iter(self.docs)


@dataclass
class QAExample:
    """One question + answer with optional gold-document context."""
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    supporting_doc_indices: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Corpus loaders
# --------------------------------------------------------------------------- #
def load_corpus(
    source: str = "auto",
    n_docs: int = 1_000,
    # The legacy `wikipedia` dataset is script-based and broken on `datasets>=2.20`.
    # `wikimedia/wikipedia` is the official parquet-based mirror with the same schema.
    hf_dataset: str = "wikimedia/wikipedia",
    hf_subset: Optional[str] = "20231101.en",
    hf_split: str = "train",
    text_field: str = "text",
    min_chars: int = 200,
    file_path: Optional[str] = None,
) -> DocumentCorpus:
    """
    Load `n_docs` text paragraphs.

    Args
    ----
    source : "auto" | "hf" | "bundled" | "file"
        auto → try hf, fall back to bundled on failure.
    n_docs : how many documents to return.
    hf_dataset, hf_subset, hf_split, text_field
        Plumbing for HuggingFace `datasets.load_dataset`.
    min_chars
        Skip very short paragraphs (the wikipedia dump has lots of titles).
    file_path
        For source="file": path to a one-paragraph-per-line text file.
    """
    if source == "auto":
        if HF_AVAILABLE:
            try:
                return _load_hf_corpus(n_docs, hf_dataset, hf_subset, hf_split,
                                      text_field, min_chars)
            except Exception as e:
                logger.warning("HF corpus load failed (%s); using bundled fallback.", e)
        return _load_bundled_corpus(n_docs)

    if source == "hf":
        if not HF_AVAILABLE:
            raise RuntimeError("source='hf' requires `pip install datasets`.")
        return _load_hf_corpus(n_docs, hf_dataset, hf_subset, hf_split,
                               text_field, min_chars)

    if source == "bundled":
        return _load_bundled_corpus(n_docs)

    if source == "file":
        if file_path is None:
            raise ValueError("source='file' requires a `file_path`.")
        return _load_file_corpus(file_path, n_docs, min_chars)

    raise ValueError(f"Unknown source {source!r}")


def _load_hf_corpus(n_docs, hf_dataset, hf_subset, hf_split, text_field, min_chars):
    """Stream from HuggingFace and stop at `n_docs`."""
    logger.info("Loading %s/%s from HuggingFace (streaming) ...", hf_dataset, hf_subset)
    ds = load_dataset(hf_dataset, hf_subset, split=hf_split, streaming=True)
    out: list[str] = []
    for row in ds:
        text = row.get(text_field)
        if not text or len(text) < min_chars:
            continue
        out.append(text)
        if len(out) >= n_docs:
            break
    return DocumentCorpus(out, source=f"hf:{hf_dataset}/{hf_subset}",
                          metadata={"split": hf_split, "min_chars": min_chars})


def _load_bundled_corpus(n_docs):
    from .fallback_sample import BUNDLED_DOCS
    if n_docs > len(BUNDLED_DOCS):
        logger.warning(
            "Bundled fallback has only %d unique paragraphs; requesting %d will "
            "repeat them %.1fx. For a credible benchmark, install HuggingFace "
            "`datasets` and re-run, or point `--source file` at a real corpus.",
            len(BUNDLED_DOCS), n_docs, n_docs / len(BUNDLED_DOCS),
        )
    docs = (BUNDLED_DOCS * ((n_docs // len(BUNDLED_DOCS)) + 1))[:n_docs]
    return DocumentCorpus(list(docs), source="bundled",
                          metadata={"unique_docs": len(BUNDLED_DOCS),
                                    "requested": n_docs})


def _load_file_corpus(file_path, n_docs, min_chars):
    out: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) < min_chars:
                continue
            out.append(line)
            if len(out) >= n_docs:
                break
    return DocumentCorpus(out, source=f"file:{file_path}",
                          metadata={"min_chars": min_chars})


# --------------------------------------------------------------------------- #
# QA dataset loaders (HF only, bundled fallback for offline smoke tests)
# --------------------------------------------------------------------------- #
# All entries are parquet-based mirrors that work with `datasets>=2.20`
# (no `trust_remote_code`, no Python dataset scripts).
_QA_SPECS = {
    "nq":       ("google-research-datasets/nq_open", None, "validation",
                 "question", "answer"),
    "hotpotqa": ("lucadiliello/hotpotqa", None, "validation",
                 "question", "answers"),
    "musique":  ("dgslibisey/MuSiQue", None, "validation",
                 "question", "answer"),
    "2wiki":    ("voidful/2WikiMultihopQA", None, "validation",
                 "question", "answer"),
}


def load_qa_dataset(
    name: str = "hotpotqa",
    n_examples: int = 100,
    source: str = "auto",
) -> list[QAExample]:
    """
    Load a few QA examples from one of the standard benchmarks.

    `source="auto"` tries HuggingFace then falls back to the bundled sample.
    The bundled sample contains 5 hand-written examples paired with
    `fallback_sample.BUNDLED_DOCS` indices, enough for an integration check.
    """
    if name not in _QA_SPECS:
        raise ValueError(f"Unknown QA dataset {name!r}. "
                         f"Choose from {sorted(_QA_SPECS)}.")
    if source == "auto":
        if HF_AVAILABLE:
            try:
                return _load_hf_qa(name, n_examples)
            except Exception as e:
                logger.warning("HF QA load failed (%s); using bundled fallback.", e)
        return _load_bundled_qa(n_examples)
    if source == "hf":
        if not HF_AVAILABLE:
            raise RuntimeError("source='hf' requires `pip install datasets`.")
        return _load_hf_qa(name, n_examples)
    if source == "bundled":
        return _load_bundled_qa(n_examples)
    raise ValueError(f"Unknown source {source!r}")


def _load_hf_qa(name: str, n_examples: int) -> list[QAExample]:
    repo, subset, split, q_field, a_field = _QA_SPECS[name]
    logger.info("Loading QA dataset %s from HF ...", name)
    ds = load_dataset(repo, subset, split=split, streaming=True)
    out: list[QAExample] = []
    for row in ds:
        question = row.get(q_field, "") or ""
        ans = row.get(a_field, None)
        if isinstance(ans, list):
            answer = ans[0] if ans else ""
        elif isinstance(ans, dict):
            answer = ans.get("text") or ans.get("answer") or ""
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
        else:
            answer = str(ans) if ans is not None else ""
        if not question or not answer:
            continue
        out.append(QAExample(question=question, answer=answer))
        if len(out) >= n_examples:
            break
    return out


def _load_bundled_qa(n_examples: int) -> list[QAExample]:
    from .fallback_sample import SAMPLE_QA_PAIRS, BUNDLED_DOCS
    out: list[QAExample] = []
    for row in SAMPLE_QA_PAIRS[:n_examples]:
        out.append(QAExample(
            question=row["question"],
            answer=row["answer"],
            contexts=[BUNDLED_DOCS[i] for i in row.get("supporting_doc_indices", [])],
            supporting_doc_indices=list(row.get("supporting_doc_indices", [])),
        ))
    return out
