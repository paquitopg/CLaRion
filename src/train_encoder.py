from __future__ import annotations

import numpy as np
from transformers import AutoTokenizer

from .data.fallback_sample import BUNDLED_DOCS, SAMPLE_QA_PAIRS
from .models.config import ModelConfig
from .models.encoder import (
    _init_params,
    build_encoder,
    save_encoder_weights,
)


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


def cosine_regression_loss_with_grad(
    pred: np.ndarray,
    target: np.ndarray,
):
    pred_n = l2_normalize(pred)
    target_n = l2_normalize(target)

    cosine = np.sum(pred_n * target_n, axis=-1)
    loss = 1.0 - float(np.mean(cosine))

    grad = (pred_n - target_n) / max(pred.shape[0], 1)
    grad = np.ascontiguousarray(grad, dtype=np.float32)
    return loss, grad


def build_doc_token_ids(docs, tokenizer, max_len):
    doc_token_ids = []
    for doc in docs:
        encoded = tokenizer(
            doc,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="np",
        )
        doc_token_ids.append(encoded["input_ids"].astype(np.int32, copy=False))
    return doc_token_ids


def run_encoder_pretrain(
    backend: str = "numpy",
    epochs: int = 2,
    lr: float = 1e-3,
    save_path: str = "./artifacts/encoder_pretrained.npz",
):
    cfg = ModelConfig(
        vocab_size=30522,
        hidden_dim=256,
        ffn_dim=512,
        n_layers=2,
        n_heads=4,
        n_memory_tokens=4,
        max_seq_len=64,
        pad_id=0,
        init_scale=0.02,
        eps=1e-6,
        rng_seed=42,
    )

    tokenizer = build_tokenizer("bert-base-uncased")
    doc_token_ids = build_doc_token_ids(BUNDLED_DOCS, tokenizer, cfg.max_seq_len)

    encoder = build_encoder(
        cfg,
        backend=backend,
        params=_init_params(cfg),
    )
    encoder.train_retrieval_head = True
    encoder.train_memory_tokens = True

    print(f"[{backend}] encoder_pretrain:start epochs={epochs}", flush=True)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for step, sample in enumerate(SAMPLE_QA_PAIRS):
            q_encoded = tokenizer(
                sample["question"],
                truncation=True,
                max_length=cfg.max_seq_len,
                padding="max_length",
                return_tensors="np",
            )
            query_ids = q_encoded["input_ids"].astype(np.int32, copy=False)

            query_vec = encoder.encode_retrieval(query_ids, pooling="mean")

            support_vecs = []
            for doc_idx in sample["supporting_doc_indices"]:
                doc_vec = encoder.encode_retrieval(doc_token_ids[doc_idx], pooling="mean")
                support_vecs.append(doc_vec)

            target_vec = np.mean(np.stack(support_vecs, axis=0), axis=0)
            target_vec = np.ascontiguousarray(target_vec, dtype=np.float32)

            loss, grad_query = cosine_regression_loss_with_grad(query_vec, target_vec)

            encoder.backward_query(
                token_ids=query_ids,
                grad_query=grad_query,
                lr=lr,
            )

            epoch_loss += float(loss)

            print(
                f"[{backend}] encoder_pretrain epoch={epoch} step={step} "
                f"loss={float(loss):.6f} "
                f"grad_norm={float(np.linalg.norm(grad_query)):.6f}",
                flush=True,
            )

        mean_loss = epoch_loss / max(len(SAMPLE_QA_PAIRS), 1)
        print(
            f"[{backend}] encoder_pretrain epoch={epoch} mean_loss={mean_loss:.6f}",
            flush=True,
        )

    save_encoder_weights(encoder, save_path)
    print(f"[{backend}] encoder_pretrain:saved path={save_path}", flush=True)


if __name__ == "__main__":
    for backend in ["cython", "numpy"]:
        run_encoder_pretrain(
            backend=backend,
            epochs=2,
            lr=1e-3,
            save_path=f"./artifacts/encoder/encoder_pretrained_{backend}.npz",
        )