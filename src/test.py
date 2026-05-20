
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from pathlib import Path
from transformers import AutoTokenizer

from clarion.models.config import ModelConfig, IndexConfig
from clarion.models.encoder import build_encoder
from clarion.models.clara_decoder import ClaraDecoder
from clarion.models.topk import ClaraTopKCython
from clarion.models.pipeline import ClaraPipeline

from clarion.index.builder import IndexBuilder
from clarion.data.fallback_sample import (
    BUNDLED_DOCS,
    SAMPLE_QA_PAIRS,
)


def build_tokenizer(name: str):

    tok = AutoTokenizer.from_pretrained(name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


def build_dataset(
    qa_pairs,
    tokenizer,
    max_len,
):

    dataset = []

    for sample in qa_pairs:

        prompt = (
            f"question: {sample['question']} answer:"
        )

        answer = sample["answer"]

        x = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        y = tokenizer(
            answer,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        dataset.append({
            "question": sample["question"],
            "answer": answer,
            "supporting_docs": sample["supporting_doc_indices"],
            "input_ids": x["input_ids"],
            "labels": y["input_ids"],
        })

    return dataset


def token_accuracy(logits, labels):

    preds = logits.argmax(dim=-1)

    mask = labels != 0

    correct = ((preds == labels) & mask).float().sum()

    total = mask.float().sum().clamp(min=1)

    return (correct / total).item()


def main():

    cfg = ModelConfig(
        vocab_size=30522,
        hidden_dim=256,
        ffn_dim=512,
        n_layers=2,
        n_heads=4,
        head_dim=64,
        n_memory_tokens=4,
        max_seq_len=64,
        max_doc_len=128,
        pad_id=0,
        init_scale=0.02,
        eps=1e-6,
        rng_seed=42,
        tokenizer_name="bert-base-uncased",
    )

    index_cfg = IndexConfig(
        index_path="./artifacts/index.npy",
        meta_path="./artifacts/index_meta.json",
        batch_size=8,
    )

    Path("./artifacts/plots").mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cpu")

    tokenizer = build_tokenizer(
        cfg.tokenizer_name
    )

    dataset = build_dataset(
        SAMPLE_QA_PAIRS,
        tokenizer,
        cfg.max_seq_len,
    )

    builder = IndexBuilder(
        model_config=cfg,
        index_config=index_cfg,
    )

    t0 = time.perf_counter()

    bank, report = builder.build(
        docs=BUNDLED_DOCS,
        backend="numpy",
        parallel=False,
        save=False,
    )

    index_time = time.perf_counter() - t0

    encoder = build_encoder(
        cfg,
        backend="numpy",
    )

    decoder = ClaraDecoder(
        d_model=cfg.hidden_dim,
        vocab_size=cfg.vocab_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
    ).to(device)

    topk = ClaraTopKCython(k=4)

    pipeline = ClaraPipeline(
        encoder=encoder,
        decoder=decoder,
        topk=topk,
    )

    losses = []
    accs = []
    retrieval_hits = []
    retrieval_positions = []
    inference_times = []

    for sample in dataset:

        input_ids = sample["input_ids"].to(device)
        labels = sample["labels"].to(device)

        t0 = time.perf_counter()

        logits, retrieved_idx = pipeline.forward(
            input_ids=input_ids,
            bank=bank,
        )

        dt = time.perf_counter() - t0

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=0,
        )

        acc = token_accuracy(
            logits[:, :-1],
            labels[:, 1:],
        )

        retrieved = (
            retrieved_idx[0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        gt_docs = sample["supporting_docs"]

        hit = any(d in retrieved for d in gt_docs)

        retrieval_hits.append(int(hit))

        pos = -1

        for i, d in enumerate(retrieved):
            if d in gt_docs:
                pos = i
                break

        retrieval_positions.append(pos)

        losses.append(loss.item())
        accs.append(acc)
        inference_times.append(dt)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker="o")
    plt.title("Loss per QA Sample")
    plt.xlabel("Sample")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./artifacts/plots/loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        np.arange(len(accs)),
        accs,
    )
    plt.title("Token Accuracy per Sample")
    plt.xlabel("Sample")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("./artifacts/plots/token_accuracy.png")
    plt.close()

    plt.figure(figsize=(6, 6))

    success = sum(retrieval_hits)
    fail = len(retrieval_hits) - success

    plt.pie(
        [success, fail],
        labels=["Hit", "Miss"],
        autopct="%1.1f%%",
    )

    plt.title("Retrieval Hit Rate")

    plt.savefig("./artifacts/plots/retrieval_hit_rate.png")
    plt.close()

    valid_pos = [x for x in retrieval_positions if x >= 0]

    plt.figure(figsize=(8, 5))

    if len(valid_pos) > 0:
        plt.hist(valid_pos, bins=np.arange(0, 6) - 0.5)

    plt.xticks(range(5))

    plt.title("Position of Correct Retrieved Doc")
    plt.xlabel("Top-K Position")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("./artifacts/plots/retrieval_positions.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(inference_times, marker="o")
    plt.title("Inference Latency")
    plt.xlabel("Sample")
    plt.ylabel("Seconds")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./artifacts/plots/inference_latency.png")
    plt.close()
    flat_bank = bank.reshape(bank.shape[0], -1)

    subset = flat_bank[:10, :64]

    plt.figure(figsize=(10, 6))
    plt.imshow(subset, aspect="auto")
    plt.colorbar()
    plt.title("Structured Memory Bank Heatmap")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig("./artifacts/plots/memory_heatmap.png")
    plt.close()

    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)

    print(f"Docs indexed        : {report.n_docs}")
    print(f"Index build time    : {index_time:.4f}s")
    print(f"Docs/sec            : {report.docs_per_s:.2f}")
    print(f"Mean loss           : {np.mean(losses):.4f}")
    print(f"Mean token accuracy : {np.mean(accs):.4f}")
    print(f"Retrieval hit rate  : {np.mean(retrieval_hits):.4f}")
    print(f"Mean inference time : {np.mean(inference_times):.4f}s")

    print("\nSaved plots:")
    print(" - loss_curve.png")
    print(" - token_accuracy.png")
    print(" - retrieval_hit_rate.png")
    print(" - retrieval_positions.png")
    print(" - inference_latency.png")
    print(" - memory_heatmap.png")


if __name__ == "__main__":
    main()
