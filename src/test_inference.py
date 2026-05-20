from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

from .data.fallback_sample import BUNDLED_DOCS, SAMPLE_QA_PAIRS
from .index.builder import IndexBuilder
from .models.config import (
    DecoderConfig,
    IndexConfig,
    ModelConfig,
    TopKConfig,
)
from .models.decoder import build_decoder, init_decoder_weights
from .models.encoder import _init_params, build_encoder
from .models.pipeline import ClaraPipeline
from .models.topk import build_topk
from .train_joint import load_full_checkpoint

sns.set_theme(style="whitegrid", font="sans-serif")


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_runtime(
    backend: str,
    checkpoint_path: str,
):
    tokenizer = build_tokenizer("bert-base-uncased")
    pad_id = int(tokenizer.pad_token_id)

    cfg = ModelConfig(
        vocab_size=int(tokenizer.vocab_size),
        hidden_dim=256,
        ffn_dim=512,
        n_layers=2,
        n_heads=4,
        n_memory_tokens=4,
        max_seq_len=64,
        pad_id=pad_id,
        init_scale=0.02,
        eps=1e-6,
        rng_seed=42,
    )

    decoder_cfg = DecoderConfig(
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_dim=cfg.ffn_dim,
        vocab_size=cfg.vocab_size,
        pad_id=cfg.pad_id,
        eps=cfg.eps,
        init_scale=cfg.init_scale,
    )

    index_cfg = IndexConfig(
        index_path="./artifacts/index.npy",
        meta_path="./artifacts/index_meta.json",
        batch_size=8,
    )

    topk_cfg = TopKConfig(
        k=4,
        temperature=1.0,
    )

    encoder = build_encoder(
        cfg,
        backend=backend,
        params=_init_params(cfg),
    )
    decoder = build_decoder(
        decoder_cfg,
        backend=backend,
        params=init_decoder_weights(decoder_cfg),
    )
    topk = build_topk(
        topk_cfg,
        backend=backend,
    )

    meta = load_full_checkpoint(
        path=checkpoint_path,
        encoder=encoder,
        decoder=decoder,
        topk=topk,
        strict=True,
    )

    builder = IndexBuilder(
        model_config=cfg,
        index_config=index_cfg,
        tokenizer_name="bert-base-uncased",
    )
    bank, report = builder.build(
        docs=BUNDLED_DOCS,
        backend=backend,
        parallel=(backend == "cython"),
        save=False,
    )

    pipeline = ClaraPipeline(
        encoder=encoder,
        decoder=decoder,
        topk=topk,
    )

    return tokenizer, pipeline, bank, meta, report


def decode_text(tokenizer, token_ids: np.ndarray) -> str:
    ids = token_ids[0].tolist()
    return tokenizer.decode(ids, skip_special_tokens=True)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def token_f1_simple(reference: str, prediction: str) -> float:
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    common = 0
    for tok, c in pred_counts.items():
        common += min(c, ref_counts.get(tok, 0))

    if common == 0:
        return 0.0

    precision = common / max(len(pred_tokens), 1)
    recall = common / max(len(ref_tokens), 1)
    return float(2.0 * precision * recall / (precision + recall))


def exact_match_text(reference: str, prediction: str) -> float:
    return float(normalize_text(reference) == normalize_text(prediction))


def retrieval_hit_at_k(retrieved_indices: list[int], supporting_doc_indices: list[int]) -> float:
    if not supporting_doc_indices:
        return np.nan
    retrieved = set(retrieved_indices)
    supporting = set(supporting_doc_indices)
    return float(len(retrieved.intersection(supporting)) > 0)


def retrieval_overlap_ratio(retrieved_indices: list[int], supporting_doc_indices: list[int]) -> float:
    if not supporting_doc_indices:
        return np.nan
    retrieved = set(retrieved_indices)
    supporting = set(supporting_doc_indices)
    return float(len(retrieved.intersection(supporting)) / max(len(supporting), 1))


def _save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_generation_examples(
    backend: str = "numpy",
    checkpoint_path: str = "./artifacts/checkpoints/clarion_full_last_numpy.npz",
    max_new_tokens: int = 24,
    temperature: float = 1.0,
    topk_sampling: int = 5,
):
    tokenizer, pipeline, bank, meta, report = build_runtime(
        backend=backend,
        checkpoint_path=checkpoint_path,
    )

    rows: list[dict] = []
    examples = SAMPLE_QA_PAIRS[: min(5, len(SAMPLE_QA_PAIRS))]

    for i, sample in enumerate(examples):
        prompt = f"question: {sample['question']} answer:"
        enc = tokenizer(
            prompt,
            truncation=True,
            max_length=64,
            padding="max_length",
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int32, copy=False)

        t0 = time.perf_counter()
        logits, retrieved = pipeline.forward(input_ids=input_ids, bank=bank)
        forward_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        generated_ids = pipeline.generate(
            input_ids=input_ids,
            bank=bank,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            topk=topk_sampling,
            eos_token_id=tokenizer.eos_token_id,
        )
        generate_time_s = time.perf_counter() - t1

        generated_text = decode_text(tokenizer, generated_ids)
        reference_answer = sample["answer"]

        retrieved_indices = retrieved[0].tolist()
        supporting_doc_indices = sample.get("supporting_doc_indices", [])

        rows.append(
            {
                "backend": backend,
                "example_id": i,
                "question": sample["question"],
                "reference_answer": reference_answer,
                "generated_text": generated_text,
                "retrieved_indices": retrieved_indices,
                "supporting_doc_indices": supporting_doc_indices,
                "forward_time_s": float(forward_time_s),
                "generate_time_s": float(generate_time_s),
                "generated_num_tokens": int(generated_ids.shape[1]),
                "exact_match_text": exact_match_text(reference_answer, generated_text),
                "token_f1_simple": token_f1_simple(reference_answer, generated_text),
                "retrieval_hit_at_k": retrieval_hit_at_k(retrieved_indices, supporting_doc_indices),
                "retrieval_overlap_ratio": retrieval_overlap_ratio(retrieved_indices, supporting_doc_indices),
            }
        )

        print(
            f"[{backend}] example={i} "
            f"forward={forward_time_s:.6f}s "
            f"generate={generate_time_s:.6f}s "
            f"f1={rows[-1]['token_f1_simple']:.4f} "
            f"hit@k={rows[-1]['retrieval_hit_at_k']}",
            flush=True,
        )

    return {
        "backend": backend,
        "rows": rows,
        "meta": {
            "backend": backend,
            "checkpoint_path": checkpoint_path,
            "checkpoint_meta": meta,
            "index_report": {
                "n_docs": getattr(report, "n_docs", None),
                "dim": getattr(report, "dim", None),
            },
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "topk_sampling": topk_sampling,
        },
    }


def save_generation_report(
    all_results: list[dict],
    output_dir: str = "./logs/inference_compare",
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [row for result in all_results for row in result["rows"]]
    df = pd.DataFrame(rows)

    metas = {result["backend"]: result["meta"] for result in all_results}
    summary_rows = []
    for backend, g in df.groupby("backend"):
        summary_rows.append(
            {
                "backend": backend,
                "n_examples": int(len(g)),
                "forward_time_mean_s": float(g["forward_time_s"].mean()),
                "generate_time_mean_s": float(g["generate_time_s"].mean()),
                "generated_num_tokens_mean": float(g["generated_num_tokens"].mean()),
                "exact_match_text_mean": float(g["exact_match_text"].mean()),
                "token_f1_simple_mean": float(g["token_f1_simple"].mean()),
                "retrieval_hit_at_k_mean": float(g["retrieval_hit_at_k"].dropna().mean()) if g["retrieval_hit_at_k"].notna().any() else np.nan,
                "retrieval_overlap_ratio_mean": float(g["retrieval_overlap_ratio"].dropna().mean()) if g["retrieval_overlap_ratio"].notna().any() else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    export_df = df.copy()
    export_df["retrieved_indices"] = export_df["retrieved_indices"].apply(json.dumps)
    export_df["supporting_doc_indices"] = export_df["supporting_doc_indices"].apply(json.dumps)

    export_df.to_csv(out / "inference_examples_compare.csv", index=False)
    summary_df.to_csv(out / "inference_summary_compare.csv", index=False)

    with (out / "inference_examples_compare.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (out / "inference_meta_compare.json").open("w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    if not df.empty:
        plot_df = df.copy()
        plot_df["generate_time_ms"] = plot_df["generate_time_s"] * 1000.0
        plot_df["forward_time_ms"] = plot_df["forward_time_s"] * 1000.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

        sns.barplot(
            data=plot_df,
            x="example_id",
            y="generate_time_ms",
            hue="backend",
            ax=axes[0],
        )
        axes[0].set_title("Generation time by example")
        axes[0].set_xlabel("Example id")
        axes[0].set_ylabel("Generate time (ms)")
        axes[0].grid(True, axis="y", alpha=0.3)

        quality_df = (
            plot_df.groupby("backend", as_index=False)[["token_f1_simple", "exact_match_text", "retrieval_hit_at_k"]]
            .mean(numeric_only=True)
            .melt(id_vars="backend", var_name="metric", value_name="value")
        )

        sns.barplot(
            data=quality_df,
            x="metric",
            y="value",
            hue="backend",
            ax=axes[1],
        )
        axes[1].set_title("Quality comparison")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.suptitle("Inference comparison", fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        _save_figure(fig, out / "inference_compare.png")

        mean_times = (
            plot_df.groupby("backend", as_index=False)[["forward_time_ms", "generate_time_ms"]]
            .mean()
            .melt(id_vars="backend", var_name="stage", value_name="time_ms")
        )

        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.barplot(
            data=mean_times,
            x="stage",
            y="time_ms",
            hue="backend",
            ax=ax,
        )
        ax.set_title("Mean inference time by backend")
        ax.set_xlabel("")
        ax.set_ylabel("Time (ms)")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        _save_figure(fig, out / "inference_time_compare.png")

    print(f"[compare] inference examples saved to {out / 'inference_examples_compare.csv'}", flush=True)
    print(f"[compare] inference summary saved to {out / 'inference_summary_compare.csv'}", flush=True)


def main():
    all_results = []

    for backend in ["cython", "numpy"]:
        result = run_generation_examples(
            backend=backend,
            checkpoint_path=f"./artifacts/checkpoints/clarion_full_last_{backend}.npz",
            max_new_tokens=24,
            temperature=1.0,
            topk_sampling=5,
        )
        all_results.append(result)

    save_generation_report(all_results, output_dir="./logs/inference_compare")


if __name__ == "__main__":
    main()