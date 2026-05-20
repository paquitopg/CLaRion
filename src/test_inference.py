from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
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


def run_generation_examples(
    backend: str = "numpy",
    checkpoint_path: str = "./artifacts/checkpoints/clarion_full_last_numpy.npz",
    max_new_tokens: int = 24,
    temperature: float = 1.0,
    topk_sampling: int = 5,
):
    output_dir = Path("./logs/inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, pipeline, bank, meta, report = build_runtime(
        backend=backend,
        checkpoint_path=checkpoint_path,
    )

    rows = []
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

        logits, retrieved = pipeline.forward(input_ids=input_ids, bank=bank)
        generated_ids = pipeline.generate(
            input_ids=input_ids,
            bank=bank,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            topk=topk_sampling,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_text = decode_text(tokenizer, generated_ids)

        rows.append(
            {
                "example_id": i,
                "question": sample["question"],
                "reference_answer": sample["answer"],
                "generated_text": generated_text,
                "retrieved_indices": json.dumps(retrieved[0].tolist()),
                "supporting_doc_indices": json.dumps(sample.get("supporting_doc_indices", [])),
            }
        )

    csv_path = output_dir / f"inference_examples_{backend}.csv"
    json_path = output_dir / f"inference_examples_{backend}.json"
    meta_path = output_dir / f"inference_meta_{backend}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example_id",
                "question",
                "reference_answer",
                "generated_text",
                "retrieved_indices",
                "supporting_doc_indices",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": backend,
                "checkpoint_path": checkpoint_path,
                "checkpoint_meta": meta,
                "index_report": {
                    "n_docs": getattr(report, "n_docs", None),
                    "dim": getattr(report, "dim", None),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[{backend}] inference examples saved to {csv_path}")
    print(f"[{backend}] inference meta saved to {meta_path}")


def main():
    for backend in ["cython", "numpy"]:
        run_generation_examples(
            backend=backend,
            checkpoint_path=f"./artifacts/checkpoints/clarion_full_last_{backend}.npz",
        )


if __name__ == "__main__":
    main()