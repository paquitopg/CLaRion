from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

from .data.fallback_sample import BUNDLED_DOCS, SAMPLE_QA_PAIRS
from .models.config import ModelConfig
from .models.encoder import _init_params, build_encoder, save_encoder_weights


sns.set_theme(style="whitegrid", font="sans-serif")


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

    def add(self, name: str, value: float):
        self.times[name].append(value)

    def to_rows(self) -> list[dict]:
        rows: list[dict] = []
        for name, values in sorted(self.times.items()):
            rows.append(
                {
                    "backend": self.backend,
                    "stage": name,
                    "count": len(values),
                    "total_s": float(np.sum(values)),
                    "mean_s": float(np.mean(values)),
                    "min_s": float(np.min(values)),
                    "max_s": float(np.max(values)),
                }
            )
        return rows


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


def cosine_regression_loss_with_grad(pred: np.ndarray, target: np.ndarray):
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


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    step_rows: list[dict] = []
    epoch_rows: list[dict] = []
    global_step = 0

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
        doc_token_ids = build_doc_token_ids(BUNDLED_DOCS, tokenizer, cfg.max_seq_len)

    with perf.track("encoder.params.init"):
        params = _init_params(cfg)

    with perf.track("encoder.build"):
        encoder = build_encoder(cfg, backend=backend, params=params)

    encoder.train_retrieval_head = True
    encoder.train_memory_tokens = True

    total_t0 = time.perf_counter()

    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        epoch_loss = 0.0
        epoch_grad_values: list[float] = []
        epoch_step_times: list[float] = []

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
            epoch_grad_values.append(grad_norm)
            epoch_step_times.append(step_dt)
            epoch_loss += loss_value

            step_rows.append(
                {
                    "backend": backend,
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss": loss_value,
                    "grad_norm": grad_norm,
                    "sample_time_s": step_dt,
                }
            )
            global_step += 1

        epoch_dt = time.perf_counter() - epoch_t0
        perf.add("epoch.total", epoch_dt)

        mean_loss = epoch_loss / max(len(SAMPLE_QA_PAIRS), 1)
        epoch_mean_losses.append(mean_loss)

        epoch_rows.append(
            {
                "backend": backend,
                "epoch": epoch,
                "epoch_mean_loss": mean_loss,
                "epoch_time_s": float(epoch_dt),
                "epoch_mean_grad_norm": float(np.mean(epoch_grad_values)) if epoch_grad_values else 0.0,
                "epoch_mean_sample_time_s": float(np.mean(epoch_step_times)) if epoch_step_times else 0.0,
            }
        )

    with perf.track("weights.save"):
        save_encoder_weights(encoder, save_path)

    total_dt = time.perf_counter() - total_t0
    perf.add("run.total", total_dt)

    summary_row = {
        "backend": backend,
        "epochs": epochs,
        "n_samples": len(SAMPLE_QA_PAIRS),
        "final_epoch_mean_loss": epoch_mean_losses[-1] if epoch_mean_losses else 0.0,
        "best_epoch_mean_loss": min(epoch_mean_losses) if epoch_mean_losses else 0.0,
        "mean_step_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "max_grad_norm": float(np.max(grad_norms)) if grad_norms else 0.0,
        "total_runtime_s": float(total_dt),
        "save_path": save_path,
    }

    return {
        "step_rows": step_rows,
        "epoch_rows": epoch_rows,
        "perf_rows": perf.to_rows(),
        "summary_row": summary_row,
    }


def save_comparison_report(all_results: list[dict], output_dir: str = "./logs/encoder_pretrain_compare"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps_df = pd.DataFrame([r for result in all_results for r in result["step_rows"]])
    epochs_df = pd.DataFrame([r for result in all_results for r in result["epoch_rows"]])
    perf_df = pd.DataFrame([r for result in all_results for r in result["perf_rows"]])
    summary_df = pd.DataFrame([result["summary_row"] for result in all_results])

    steps_df.to_csv(out / "steps_compare.csv", index=False)
    epochs_df.to_csv(out / "epochs_compare.csv", index=False)
    perf_df.to_csv(out / "perf_compare.csv", index=False)
    summary_df.to_csv(out / "summary_compare.csv", index=False)

    if not steps_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

        loss_df = (
            steps_df.groupby("global_step", as_index=False)["loss"]
            .mean()
            .sort_values("global_step")
        )

        sns.lineplot(
            data=loss_df,
            x="global_step",
            y="loss",
            color="mediumpurple",
            marker="o",
            ax=axes[0],
        )
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Global step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        plot_steps = steps_df.copy().sort_values(["backend", "global_step"])
        plot_steps["sample_time_ms"] = plot_steps["sample_time_s"] * 1000.0
        plot_steps["sample_time_ms_smooth"] = (
            plot_steps.groupby("backend")["sample_time_ms"]
            .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        )

        sns.lineplot(
            data=plot_steps,
            x="global_step",
            y="sample_time_ms_smooth",
            hue="backend",
            marker="o",
            ax=axes[1],
        )
        axes[1].set_title("Sample time comparison")
        axes[1].set_xlabel("Global step")
        axes[1].set_ylabel("Sample time (ms)")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Encoder pretrain comparison", fontsize=18)
        fig.tight_layout()
        _save_fig(fig, out / "backend_compare.png")

    if not perf_df.empty:
        threshold = 0.01
        perf_plot = perf_df[perf_df["mean_s"] >= threshold].copy()
        perf_plot = perf_plot.sort_values(["mean_s", "backend"], ascending=[False, True])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=perf_plot.sort_values(["stage", "backend"]),
            y="stage",
            x="mean_s",
            hue="backend",
            ax=ax,
            orient="h",
        )
        ax.set_title("Mean stage time comparison")
        ax.set_xlabel("Mean time (s)")
        ax.set_ylabel("Stage")
        _save_fig(fig, out / "perf_stage_compare.png")


if __name__ == "__main__":
    results = []
    for backend in ["cython", "numpy"]:
        results.append(
            run_encoder_pretrain(
                backend=backend,
                epochs=2,
                lr=1e-3,
                save_path=f"./artifacts/encoder/encoder_pretrained_{backend}.npz",
            )
        )

    save_comparison_report(results)