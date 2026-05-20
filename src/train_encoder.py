from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .data.fallback_sample import BUNDLED_DOCS, SAMPLE_QA_PAIRS
from .models.config import ModelConfig
from .models.encoder import (
    _init_params,
    build_encoder,
    save_encoder_weights,
)


class PerfTracker:
    def __init__(self, backend: str):
        self.backend = backend
        self.times = defaultdict(list)

    @contextmanager
    def track(self, name: str):
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.times[name].append(dt)
        print(f"[{self.backend}] {name}: {dt:.6f}s", flush=True)

    def add(self, name: str, value: float):
        self.times[name].append(value)

    def summary(self):
        print(f"\n[{self.backend}] ===== PRETRAIN PERF SUMMARY =====", flush=True)
        for name, values in sorted(self.times.items()):
            total = float(np.sum(values))
            mean = float(np.mean(values))
            count = len(values)
            print(
                f"[{self.backend}] {name:<30} "
                f"count={count:<4d} total={total:>10.6f}s mean={mean:>10.6f}s",
                flush=True,
            )
        print(f"[{self.backend}] =================================\n", flush=True)


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


def build_doc_token_ids(docs, tokenizer, max_len, perf: PerfTracker | None = None):
    doc_token_ids = []

    for i, doc in enumerate(docs):
        if perf is None:
            encoded = tokenizer(
                doc,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="np",
            )
        else:
            with perf.track("docs.tokenize"):
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
    perf = PerfTracker(backend)
    losses: list[float] = []
    grad_norms: list[float] = []
    epoch_mean_losses: list[float] = []

    with perf.track("config.build"):
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

    with perf.track("tokenizer.build"):
        tokenizer = build_tokenizer("bert-base-uncased")

    with perf.track("docs.build_token_ids"):
        doc_token_ids = build_doc_token_ids(
            BUNDLED_DOCS,
            tokenizer,
            cfg.max_seq_len,
            perf=None,
        )

    with perf.track("encoder.params.init"):
        params = _init_params(cfg)

    with perf.track("encoder.build"):
        encoder = build_encoder(
            cfg,
            backend=backend,
            params=params,
        )

    encoder.train_retrieval_head = True
    encoder.train_memory_tokens = True

    print(f"[{backend}] encoder_pretrain:start epochs={epochs}", flush=True)

    total_t0 = time.perf_counter()

    for epoch in range(epochs):
        print(f"[{backend}] encoder_pretrain epoch={epoch}:start", flush=True)
        epoch_t0 = time.perf_counter()
        epoch_loss = 0.0

        for step, sample in enumerate(SAMPLE_QA_PAIRS):
            step_t0 = time.perf_counter()

            with perf.track("sample.tokenize.query"):
                q_encoded = tokenizer(
                    sample["question"],
                    truncation=True,
                    max_length=cfg.max_seq_len,
                    padding="max_length",
                    return_tensors="np",
                )
                query_ids = q_encoded["input_ids"].astype(np.int32, copy=False)

            with perf.track("sample.encode.query"):
                query_vec = encoder.encode_retrieval(query_ids, pooling="mean")

            support_vecs = []
            for doc_idx in sample["supporting_doc_indices"]:
                with perf.track("sample.encode.support_doc"):
                    doc_vec = encoder.encode_retrieval(
                        doc_token_ids[doc_idx],
                        pooling="mean",
                    )
                support_vecs.append(doc_vec)

            with perf.track("sample.target.build"):
                target_vec = np.mean(np.stack(support_vecs, axis=0), axis=0)
                target_vec = np.ascontiguousarray(target_vec, dtype=np.float32)

            with perf.track("sample.loss"):
                loss, grad_query = cosine_regression_loss_with_grad(query_vec, target_vec)

            with perf.track("sample.backward"):
                encoder.backward_query(
                    token_ids=query_ids,
                    grad_query=grad_query,
                    lr=lr,
                )

            step_dt = time.perf_counter() - step_t0
            perf.add("sample.total", step_dt)

            loss_value = float(loss)
            grad_norm = float(np.linalg.norm(grad_query))

            losses.append(loss_value)
            grad_norms.append(grad_norm)
            epoch_loss += loss_value

            print(
                f"[{backend}] encoder_pretrain epoch={epoch} step={step} "
                f"loss={loss_value:.6f} "
                f"grad_norm={grad_norm:.6f} "
                f"sample_time={step_dt:.6f}s",
                flush=True,
            )

        epoch_dt = time.perf_counter() - epoch_t0
        perf.add("epoch.total", epoch_dt)

        mean_loss = epoch_loss / max(len(SAMPLE_QA_PAIRS), 1)
        epoch_mean_losses.append(mean_loss)

        print(
            f"[{backend}] encoder_pretrain epoch={epoch} "
            f"mean_loss={mean_loss:.6f} "
            f"epoch_time={epoch_dt:.6f}s",
            flush=True,
        )

    with perf.track("weights.save"):
        save_encoder_weights(encoder, save_path)

    total_dt = time.perf_counter() - total_t0
    perf.add("run.total", total_dt)

    print(f"[{backend}] encoder_pretrain:saved path={save_path}", flush=True)

    print(f"\n[{backend}] ===== PRETRAIN METRICS SUMMARY =====", flush=True)
    print(f"[{backend}] epochs={epochs}", flush=True)
    print(f"[{backend}] n_samples={len(SAMPLE_QA_PAIRS)}", flush=True)
    print(f"[{backend}] final_epoch_mean_loss={epoch_mean_losses[-1]:.6f}", flush=True)
    print(f"[{backend}] best_epoch_mean_loss={min(epoch_mean_losses):.6f}", flush=True)
    print(f"[{backend}] mean_step_loss={float(np.mean(losses)):.6f}", flush=True)
    print(f"[{backend}] mean_grad_norm={float(np.mean(grad_norms)):.6f}", flush=True)
    print(f"[{backend}] max_grad_norm={float(np.max(grad_norms)):.6f}", flush=True)
    print(f"[{backend}] total_runtime={total_dt:.6f}s", flush=True)
    print(f"[{backend}] ==================================\n", flush=True)

    perf.summary()


if __name__ == "__main__":
    for backend in ["cython", "numpy"]:
        run_encoder_pretrain(
            backend=backend,
            epochs=2,
            lr=1e-3,
            save_path=f"./artifacts/encoder/encoder_pretrained_{backend}.npz",
        )