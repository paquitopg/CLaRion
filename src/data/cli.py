"""
CLaRiON data ingestion CLI.

Examples:

    # Materialize 10k Wikipedia paragraphs to disk (uses HF datasets).
    python -m src.data.cli pull-wiki --n 10000 --out data/wiki_10k.txt

    # Pull 500 HotpotQA validation examples.
    python -m src.data.cli pull-qa --name hotpotqa --n 500 --out data/hotpotqa.jsonl

    # Run the full encoder-index benchmark on a real corpus instead of
    # the synthetic Zipf generator.
    python -m src.data.cli bench-on-real --n-docs 5000

Once a corpus file exists, point the index builder at it directly:

    from src.data import load_corpus
    from src.index.builder import IndexBuilder
    corpus = load_corpus(source="file", file_path="data/wiki_10k.txt",
                         n_docs=5000)
    bank, report = IndexBuilder(model_cfg, index_cfg).build(corpus.docs,
                                                            backend="cython")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .loaders import load_corpus, load_qa_dataset, HF_AVAILABLE

logger = logging.getLogger("clarion.data.cli")
logging.basicConfig(level=logging.INFO,
                    format="[%(name)s] %(levelname)s: %(message)s")


def cmd_pull_wiki(args) -> int:
    corpus = load_corpus(
        source=args.source, n_docs=args.n,
        hf_dataset=args.hf_dataset, hf_subset=args.hf_subset,
        hf_split=args.hf_split, min_chars=args.min_chars,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for doc in corpus.docs:
            # Newlines inside paragraphs would break our 1-line-per-doc format.
            f.write(" ".join(doc.split()) + "\n")
    logger.info("Wrote %d docs from %s to %s", len(corpus), corpus.source, args.out)
    return 0


def cmd_pull_qa(args) -> int:
    examples = load_qa_dataset(name=args.name, n_examples=args.n, source=args.source)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "question": ex.question, "answer": ex.answer,
                "contexts": ex.contexts,
                "supporting_doc_indices": ex.supporting_doc_indices,
            }) + "\n")
    logger.info("Wrote %d QA examples (%s) to %s", len(examples), args.name, args.out)
    return 0


def cmd_bench_on_real(args) -> int:
    """Re-run the existing bench_pipeline harness but on a real corpus."""
    from ..models.config import ModelConfig
    from ..models.encoder import build_encoder
    from ..index.builder import (
        encode_corpus_parallel, encode_corpus_serial, tokenize_corpus,
    )
    from ..index.scorer import cosine_cython_omp, cosine_numpy, top_k_indices
    from ..benchmarks._timing import TimingResult, print_table, save_results, time_call

    corpus = load_corpus(source=args.source, n_docs=args.n_docs,
                        file_path=args.file_path)
    print(f"\nCorpus: {len(corpus)} docs from {corpus.source}")

    cfg = ModelConfig(hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
                      n_memory_tokens=4, max_doc_len=128, vocab_size=8_000)
    base = build_encoder(cfg, backend="numpy")
    params = base.params

    print("Tokenizing ...")
    tok = tokenize_corpus(corpus.docs, cfg)

    rows: list[TimingResult] = []
    rows.append(time_call(
        lambda: encode_corpus_serial(tok, base, batch_size=64),
        warmup=1, iters=2, label="encode numpy serial",
    ))
    for nt in (1, 2, 4):
        cy = build_encoder(cfg, backend="cython", num_threads=nt, params=params)
        if not cy.available:
            print("(cython unavailable, skipping)")
            break
        rows.append(time_call(
            lambda enc=cy: encode_corpus_serial(tok, enc, batch_size=64),
            warmup=1, iters=2, label=f"encode cython(nt={nt})",
        ))

    print(f"\n== Phase A: offline encode on real corpus, N={len(corpus)} ==")
    print_table(rows, baseline_label=None)
    save_results(rows, args.out)
    print(f"\nSaved JSON to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="src.data.cli")
    p.add_argument("--quiet", action="store_true")
    subs = p.add_subparsers(dest="cmd", required=True)

    p_w = subs.add_parser("pull-wiki", help="Materialize Wikipedia paragraphs to a file.")
    p_w.add_argument("--source", default="auto", choices=("auto", "hf", "bundled"))
    p_w.add_argument("--n", type=int, default=10_000)
    p_w.add_argument("--hf-dataset", default="wikimedia/wikipedia")
    p_w.add_argument("--hf-subset", default="20231101.en")
    p_w.add_argument("--hf-split", default="train")
    p_w.add_argument("--min-chars", type=int, default=200)
    p_w.add_argument("--out", default="data/wiki.txt")
    p_w.set_defaults(func=cmd_pull_wiki)

    p_q = subs.add_parser("pull-qa", help="Materialize a QA dataset to JSONL.")
    p_q.add_argument("--name", default="hotpotqa",
                     choices=("nq", "hotpotqa", "musique", "2wiki"))
    p_q.add_argument("--n", type=int, default=500)
    p_q.add_argument("--source", default="auto", choices=("auto", "hf", "bundled"))
    p_q.add_argument("--out", default="data/qa.jsonl")
    p_q.set_defaults(func=cmd_pull_qa)

    p_b = subs.add_parser("bench-on-real",
                          help="Run encoder benchmark on a real corpus.")
    p_b.add_argument("--source", default="auto", choices=("auto", "hf", "bundled", "file"))
    p_b.add_argument("--file-path", default=None)
    p_b.add_argument("--n-docs", type=int, default=2_000)
    p_b.add_argument("--out", default="reports/encoder_bench_real.json")
    p_b.set_defaults(func=cmd_bench_on_real)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
