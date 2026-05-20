from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .data.fallback_sample import (
    BUNDLED_DOCS,
    SAMPLE_QA_PAIRS,
)
from .index.builder import IndexBuilder
from .models.decoder import (
    build_decoder,
    init_decoder_weights,
    decoder_state_dict,
    load_decoder_state_dict,
)
from .models.config import (
    DecoderConfig,
    IndexConfig,
    LossConfig,
    ModelConfig,
    TopKConfig,
)
from .models.encoder import (
    _init_params,
    build_encoder,
    load_encoder_weights,
)
from .models.loss import clara_lm_loss, cross_entropy_with_grad
from .models.pipeline import ClaraPipeline
from .models.topk import (
    build_topk,
    topk_state_dict,
    load_topk_state_dict,
)

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

    def summary(self):
        print(f"\n[{self.backend}] ===== PERF SUMMARY =====", flush=True)
        for name, values in sorted(self.times.items()):
            total = float(np.sum(values))
            mean = float(np.mean(values))
            count = len(values)
            print(
                f"[{self.backend}] {name:<28} "
                f"count={count:<4d} total={total:>10.6f}s mean={mean:>10.6f}s",
                flush=True,
            )
        print(f"[{self.backend}] ========================\n", flush=True)

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


def token_accuracy(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> float:
    preds = np.argmax(logits, axis=-1)
    mask = labels != ignore_index
    correct = ((preds == labels) & mask).sum()
    total = np.maximum(mask.sum(), 1)
    return float(correct / total)


def exact_match(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> float:
    preds = np.argmax(logits, axis=-1)
    mask = labels != ignore_index

    row_ok = []
    for p_row, y_row, m_row in zip(preds, labels, mask):
        valid = m_row.astype(bool)
        if valid.sum() == 0:
            row_ok.append(1.0)
        else:
            row_ok.append(float(np.all(p_row[valid] == y_row[valid])))

    return float(np.mean(row_ok))


def evaluate(
    pipeline: ClaraPipeline,
    bank: np.ndarray,
    data: list[tuple[np.ndarray, np.ndarray]],
    loss_backend: str = "numpy",
    ignore_index: int = 0,
    perf: PerfTracker | None = None,
):
    losses = []
    accs = []
    ems = []

    loss_cfg = LossConfig(ignore_index=ignore_index, num_threads=0)

    for input_ids, labels in data:
        if perf is None:
            logits, _ = pipeline.forward(input_ids=input_ids, bank=bank)
            loss = clara_lm_loss(
                logits,
                labels,
                config=loss_cfg,
                backend=loss_backend,
            )
        else:
            with perf.track("eval.forward"):
                logits, _ = pipeline.forward(input_ids=input_ids, bank=bank)

            with perf.track("eval.loss"):
                loss = clara_lm_loss(
                    logits,
                    labels,
                    config=loss_cfg,
                    backend=loss_backend,
                )

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        losses.append(float(loss))
        accs.append(token_accuracy(shifted_logits, shifted_labels, ignore_index))
        ems.append(exact_match(shifted_logits, shifted_labels, ignore_index))

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "token_acc": float(np.mean(accs)) if accs else 0.0,
        "exact_match": float(np.mean(ems)) if ems else 0.0,
    }


def build_dataset(
    qa_pairs: list[dict],
    tokenizer,
    max_len: int,
    perf: PerfTracker,
):
    dataset = []

    for s in qa_pairs:
        with perf.track("dataset.tokenize.prompt"):
            x = tokenizer(
                f"question: {s['question']} answer:",
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="np",
            )

        with perf.track("dataset.tokenize.answer"):
            y = tokenizer(
                s["answer"],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="np",
            )

        with perf.track("dataset.astype"):
            dataset.append(
                (
                    x["input_ids"].astype(np.int32, copy=False),
                    y["input_ids"].astype(np.int32, copy=False),
                )
            )

    return dataset


def maybe_load_encoder_pretrain(
    encoder,
    pretrained_encoder_path: str | None,
    backend: str,
) -> bool:
    if pretrained_encoder_path is None:
        print(f"[{backend}] encoder_pretrain:skip no path provided", flush=True)
        return False

    path = Path(pretrained_encoder_path)
    if not path.exists():
        print(
            f"[{backend}] encoder_pretrain:skip missing path={path}",
            flush=True,
        )
        return False

    load_encoder_weights(encoder, str(path))
    print(
        f"[{backend}] encoder_pretrain:loaded path={path}",
        flush=True,
    )
    return True


def _encoder_state_dict(encoder) -> dict[str, np.ndarray]:
    p = encoder.params
    arrays: dict[str, np.ndarray] = {
        "encoder.embed": np.asarray(p.embed, dtype=np.float32),
        "encoder.pos_embed": np.asarray(p.pos_embed, dtype=np.float32),
        "encoder.norm_final": np.asarray(p.norm_final, dtype=np.float32),
        "encoder.memory": np.asarray(p.memory.weights, dtype=np.float32),
        "encoder.retrieval_proj": np.asarray(encoder.retrieval_proj, dtype=np.float32),
        "encoder.query_bias": np.asarray(encoder.query_bias, dtype=np.float32),
        "encoder.n_layers": np.asarray([len(p.layers)], dtype=np.int32),
    }

    for i, layer in enumerate(p.layers):
        arrays[f"encoder.layers.{i}.Wq"] = np.asarray(layer.Wq, dtype=np.float32)
        arrays[f"encoder.layers.{i}.Wk"] = np.asarray(layer.Wk, dtype=np.float32)
        arrays[f"encoder.layers.{i}.Wv"] = np.asarray(layer.Wv, dtype=np.float32)
        arrays[f"encoder.layers.{i}.Wo"] = np.asarray(layer.Wo, dtype=np.float32)
        arrays[f"encoder.layers.{i}.W1"] = np.asarray(layer.W1, dtype=np.float32)
        arrays[f"encoder.layers.{i}.W2"] = np.asarray(layer.W2, dtype=np.float32)
        arrays[f"encoder.layers.{i}.norm1"] = np.asarray(layer.norm1, dtype=np.float32)
        arrays[f"encoder.layers.{i}.norm2"] = np.asarray(layer.norm2, dtype=np.float32)

    return arrays


def _load_encoder_from_full_checkpoint(encoder, ckpt) -> None:
    p = encoder.params

    n_layers_ckpt = int(ckpt["encoder.n_layers"][0])
    if n_layers_ckpt != len(p.layers):
        raise ValueError(
            f"Encoder layer mismatch: checkpoint={n_layers_ckpt}, model={len(p.layers)}"
        )

    p.embed[...] = ckpt["encoder.embed"]
    p.pos_embed[...] = ckpt["encoder.pos_embed"]
    p.norm_final[...] = ckpt["encoder.norm_final"]
    p.memory.weights[...] = ckpt["encoder.memory"]

    encoder.retrieval_proj[...] = ckpt["encoder.retrieval_proj"]
    encoder.query_bias[...] = ckpt["encoder.query_bias"]

    for i, layer in enumerate(p.layers):
        layer.Wq[...] = ckpt[f"encoder.layers.{i}.Wq"]
        layer.Wk[...] = ckpt[f"encoder.layers.{i}.Wk"]
        layer.Wv[...] = ckpt[f"encoder.layers.{i}.Wv"]
        layer.Wo[...] = ckpt[f"encoder.layers.{i}.Wo"]
        layer.W1[...] = ckpt[f"encoder.layers.{i}.W1"]
        layer.W2[...] = ckpt[f"encoder.layers.{i}.W2"]
        layer.norm1[...] = ckpt[f"encoder.layers.{i}.norm1"]
        layer.norm2[...] = ckpt[f"encoder.layers.{i}.norm2"]

    if hasattr(encoder, "_prepare_contiguous_params"):
        encoder._params_prepared = False
        encoder._prepare_contiguous_params()


def save_full_checkpoint(
    path: str,
    encoder,
    decoder,
    topk,
    config: ModelConfig,
    decoder_config: DecoderConfig,
    topk_config: TopKConfig,
    backend: str,
    epoch: int,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    arrays.update(_encoder_state_dict(encoder))
    arrays.update(decoder_state_dict(decoder))
    arrays.update(topk_state_dict(topk))

    arrays["meta_epoch"] = np.asarray([epoch], dtype=np.int32)
    arrays["meta_backend"] = np.asarray([backend], dtype="<U16")

    arrays["meta_hidden_dim"] = np.asarray([config.hidden_dim], dtype=np.int32)
    arrays["meta_ffn_dim"] = np.asarray([config.ffn_dim], dtype=np.int32)
    arrays["meta_n_layers"] = np.asarray([config.n_layers], dtype=np.int32)
    arrays["meta_n_heads"] = np.asarray([config.n_heads], dtype=np.int32)
    arrays["meta_n_memory_tokens"] = np.asarray([config.n_memory_tokens], dtype=np.int32)
    arrays["meta_max_seq_len"] = np.asarray([config.max_seq_len], dtype=np.int32)
    arrays["meta_vocab_size"] = np.asarray([config.vocab_size], dtype=np.int32)
    arrays["meta_pad_id"] = np.asarray([config.pad_id], dtype=np.int32)
    arrays["meta_eps"] = np.asarray([config.eps], dtype=np.float32)

    arrays["meta_decoder_hidden_dim"] = np.asarray([decoder_config.hidden_dim], dtype=np.int32)
    arrays["meta_decoder_ffn_dim"] = np.asarray([decoder_config.ffn_dim], dtype=np.int32)
    arrays["meta_decoder_n_layers"] = np.asarray([decoder_config.n_layers], dtype=np.int32)
    arrays["meta_decoder_n_heads"] = np.asarray([decoder_config.n_heads], dtype=np.int32)

    arrays["meta_topk_k"] = np.asarray([topk_config.k], dtype=np.int32)
    arrays["meta_topk_temperature"] = np.asarray([topk_config.temperature], dtype=np.float32)

    np.savez_compressed(path_obj, **arrays)


def load_full_checkpoint(
    path: str,
    encoder,
    decoder,
    topk,
    strict: bool = True,
) -> dict[str, int | float | str]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path_obj}")

    with np.load(path_obj, allow_pickle=False) as ckpt:
        _load_encoder_from_full_checkpoint(encoder, ckpt)
        load_decoder_state_dict(decoder, ckpt)
        load_topk_state_dict(topk, ckpt, strict=strict)

        meta = {
            "epoch": int(ckpt["meta_epoch"][0]) if "meta_epoch" in ckpt else -1,
            "backend": str(ckpt["meta_backend"][0]) if "meta_backend" in ckpt else "unknown",
            "hidden_dim": int(ckpt["meta_hidden_dim"][0]) if "meta_hidden_dim" in ckpt else -1,
            "decoder_hidden_dim": int(ckpt["meta_decoder_hidden_dim"][0]) if "meta_decoder_hidden_dim" in ckpt else -1,
            "topk_k": int(ckpt["meta_topk_k"][0]) if "meta_topk_k" in ckpt else -1,
            "topk_temperature": float(ckpt["meta_topk_temperature"][0]) if "meta_topk_temperature" in ckpt else -1.0,
        }

    return meta


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_comparison_report(all_results: list[dict], output_dir: str = "./logs/clarion_experiment_compare"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_steps_df = pd.DataFrame([r for result in all_results for r in result["train_step_rows"]])
    epochs_df = pd.DataFrame([r for result in all_results for r in result["epoch_rows"]])
    perf_df = pd.DataFrame([r for result in all_results for r in result["perf_rows"]])
    summary_df = pd.DataFrame([result["summary_row"] for result in all_results])

    train_steps_df.to_csv(out / "train_steps_compare.csv", index=False)
    epochs_df.to_csv(out / "epochs_compare.csv", index=False)
    perf_df.to_csv(out / "perf_compare.csv", index=False)
    summary_df.to_csv(out / "summary_compare.csv", index=False)

    if not train_steps_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

        loss_df = (
            train_steps_df.groupby("global_step", as_index=False)["train_loss"]
            .mean()
            .sort_values("global_step")
        )

        sns.lineplot(
            data=loss_df,
            x="global_step",
            y="train_loss",
            color="mediumpurple",
            ax=axes[0],
        )
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Global step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        plot_steps = train_steps_df.copy().sort_values(["backend", "global_step"])
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
            ax=axes[1],
        )
        axes[1].set_title("Train sample time comparison")
        axes[1].set_xlabel("Global step")
        axes[1].set_ylabel("Sample time (ms)")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Clara experiment comparison", fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        _save_fig(fig, out / "experiment_compare.png")

    if not perf_df.empty:
        perf_plot = perf_df.sort_values(["mean_s", "backend"], ascending=[False, True])

        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(
            data=perf_plot,
            y="stage",
            x="mean_s",
            hue="backend",
            orient="h",
            ax=ax,
        )
        ax.set_title("Mean stage time by backend")
        ax.set_xlabel("Mean time (s)")
        ax.set_ylabel("Stage")
        ax.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        _save_fig(fig, out / "stage_time_compare.png")


def run_experiment(
    backend: str,
    epochs: int = 2,
    lr_decoder: float = 1e-3,
    lr_query: float = 1e-3,
    pretrained_encoder_path: str | None = None,
):
    perf = PerfTracker(backend)
    print(f"[{backend}] === run_experiment:start ===", flush=True)

    loss_backend = backend
    train_step_rows: list[dict] = []
    epoch_rows: list[dict] = []

    with perf.track("tokenizer.build"):
        tokenizer = build_tokenizer("bert-base-uncased")

    ignore_index = int(tokenizer.pad_token_id)

    with perf.track("config.build"):
        cfg = ModelConfig(
            vocab_size=int(tokenizer.vocab_size),
            hidden_dim=256,
            ffn_dim=512,
            n_layers=2,
            n_heads=4,
            n_memory_tokens=4,
            max_seq_len=64,
            pad_id=ignore_index,
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

    with perf.track("dataset.build.total"):
        dataset = build_dataset(SAMPLE_QA_PAIRS, tokenizer, cfg.max_seq_len, perf)

    with perf.track("dataset.split"):
        train_data, temp = train_test_split(
            dataset,
            test_size=0.3,
            random_state=42,
        )
        dev_data, test_data = train_test_split(
            temp,
            test_size=0.5,
            random_state=42,
        )

    with perf.track("index_builder.init"):
        builder = IndexBuilder(
            model_config=cfg,
            index_config=index_cfg,
            tokenizer_name="bert-base-uncased",
        )

    with perf.track("index_builder.build"):
        bank, report = builder.build(
            docs=BUNDLED_DOCS,
            backend=backend,
            parallel=(backend == "cython"),
            save=False,
        )

    n_docs = getattr(report, "n_docs", "na")
    dim = getattr(report, "dim", "na")

    print(
        f"[{backend}] index report: n_docs={n_docs} dim={dim}",
        flush=True,
    )

    with perf.track("encoder.params.init"):
        encoder_params = _init_params(cfg)

    with perf.track("decoder.params.init"):
        decoder_params = init_decoder_weights(decoder_cfg)

    with perf.track("encoder.build"):
        encoder = build_encoder(
            cfg,
            backend=backend,
            params=encoder_params,
        )

    with perf.track("encoder.pretrain.load"):
        maybe_load_encoder_pretrain(
            encoder=encoder,
            pretrained_encoder_path=pretrained_encoder_path,
            backend=backend,
        )

    with perf.track("decoder.build"):
        decoder = build_decoder(
            decoder_cfg,
            backend=backend,
            params=decoder_params,
        )

    with perf.track("topk.build"):
        topk = build_topk(
            topk_cfg,
            backend=backend,
        )

    with perf.track("pipeline.init"):
        pipeline = ClaraPipeline(
            encoder=encoder,
            decoder=decoder,
            topk=topk,
        )

    ckpt_dir = Path("./artifacts/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / f"clarion_full_last_{backend}.npz"

    total_train_loss = 0.0
    global_step = 0

    print(f"[{backend}] train:start epochs={epochs}", flush=True)
    for epoch in range(epochs):
        print(f"[{backend}] epoch={epoch}:start", flush=True)
        epoch_t0 = time.perf_counter()

        for step, (input_ids, labels) in enumerate(train_data):
            sample_t0 = time.perf_counter()

            with perf.track("train.forward"):
                logits, indices = pipeline.forward(
                    input_ids=input_ids,
                    bank=bank,
                )

            with perf.track("train.loss"):
                loss, grad_logits = cross_entropy_with_grad(
                    logits,
                    labels,
                    ignore_index=ignore_index,
                    backend=loss_backend,
                )

            with perf.track("train.backward.decoder"):
                grad_memory = decoder.backward(
                    grad_logits,
                    lr=lr_decoder,
                    return_grad_memory=True,
                )

            with perf.track("train.backward.retrieval"):
                grad_query = pipeline.backward(
                    input_ids=input_ids,
                    grad_memory=grad_memory,
                    lr=lr_query,
                )

            sample_dt = time.perf_counter() - sample_t0
            total_train_loss += float(loss)
            perf.add("train.sample.total", sample_dt)

            query_grad_norm = float(np.linalg.norm(grad_query))

            train_step_rows.append(
                {
                    "backend": backend,
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "train_loss": float(loss),
                    "query_grad_norm": query_grad_norm,
                    "sample_time_s": float(sample_dt),
                }
            )
            global_step += 1

            print(
                f"[{backend}] epoch={epoch} step={step} "
                f"loss={float(loss):.6f} "
                f"query_grad_norm={query_grad_norm:.6f} "
                f"topk_shape={tuple(indices.shape)} "
                f"sample_time={sample_dt:.6f}s",
                flush=True,
            )

        epoch_dt = time.perf_counter() - epoch_t0
        perf.add("train.epoch.total", epoch_dt)

        with perf.track("dev.evaluate.total"):
            metrics = evaluate(
                pipeline=pipeline,
                bank=bank,
                data=dev_data,
                loss_backend=loss_backend,
                ignore_index=ignore_index,
                perf=perf,
            )

        epoch_rows.append(
            {
                "backend": backend,
                "epoch": epoch,
                "dev_loss": float(metrics["loss"]),
                "dev_token_acc": float(metrics["token_acc"]),
                "dev_exact_match": float(metrics["exact_match"]),
                "train_epoch_time_s": float(epoch_dt),
            }
        )

        print(
            f"[{backend}] epoch={epoch}:done "
            f"loss={metrics['loss']:.4f} "
            f"token_acc={metrics['token_acc']:.4f} "
            f"exact_match={metrics['exact_match']:.4f}",
            flush=True,
        )

        save_full_checkpoint(
            path=str(last_ckpt_path),
            encoder=encoder,
            decoder=decoder,
            topk=topk,
            config=cfg,
            decoder_config=decoder_cfg,
            topk_config=topk_cfg,
            backend=backend,
            epoch=epoch,
        )
        print(
            f"[{backend}] checkpoint:last path={last_ckpt_path}",
            flush=True,
        )

    with perf.track("test.evaluate.total"):
        test_metrics = evaluate(
            pipeline=pipeline,
            bank=bank,
            data=test_data,
            loss_backend=loss_backend,
            ignore_index=ignore_index,
            perf=perf,
        )

    print(
        f"[{backend}] test "
        f"loss={test_metrics['loss']:.4f} "
        f"token_acc={test_metrics['token_acc']:.4f} "
        f"exact_match={test_metrics['exact_match']:.4f}",
        flush=True,
    )

    print(f"[{backend}] total_train_loss={total_train_loss:.6f}", flush=True)
    perf.summary()
    print(f"[{backend}] === run_experiment:end ===", flush=True)

    return {
        "train_step_rows": train_step_rows,
        "epoch_rows": epoch_rows,
        "perf_rows": perf.to_rows(),
        "summary_row": {
            "backend": backend,
            "epochs": epochs,
            "total_train_loss": float(total_train_loss),
            "test_loss": float(test_metrics["loss"]),
            "test_token_acc": float(test_metrics["token_acc"]),
            "test_exact_match": float(test_metrics["exact_match"]),
            "checkpoint_path": str(last_ckpt_path),
        },
    }


def main():
    results = []

    for backend in ["cython", "numpy"]:
        print("\n==============================")
        print(f"RUNNING BACKEND = {backend}")
        print("==============================\n")

        result = run_experiment(
            backend=backend,
            epochs=2,
            lr_decoder=1e-3,
            lr_query=1e-3,
            pretrained_encoder_path=f"./artifacts/encoder/encoder_pretrained_{backend}.npz",
        )
        results.append(result)

    save_comparison_report(results)


if __name__ == "__main__":
    main()