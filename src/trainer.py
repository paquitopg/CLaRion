import time
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .models.config import (
    ModelConfig,
    IndexConfig,
    DecoderConfig,
    TopKConfig,
    LossConfig,
)
from .models.encoder import build_encoder, _init_params
from .models.clara_decoder import build_decoder, init_decoder_weights
from .models.topk import build_topk
from .models.pipeline import ClaraPipeline
from .models.loss import cross_entropy_with_grad, clara_lm_loss
from .index.builder import IndexBuilder
from .data.fallback_sample import (
    BUNDLED_DOCS,
    SAMPLE_QA_PAIRS,
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


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def token_accuracy(logits, labels):
    preds = np.argmax(logits, axis=-1)
    mask = labels != 0
    correct = ((preds == labels) & mask).sum()
    total = np.maximum(mask.sum(), 1)
    return float(correct / total)


def exact_match(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return float(np.all(preds == labels, axis=-1).mean())


def evaluate(
    pipeline,
    bank,
    data,
    loss_backend: str = "numpy",
    ignore_index: int = 0,
    perf: PerfTracker | None = None,
):
    losses = []
    accs = []
    ems = []

    loss_cfg = LossConfig(ignore_index=ignore_index, num_threads=0)

    for step, (input_ids, labels) in enumerate(data):
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

        losses.append(float(loss))
        accs.append(token_accuracy(logits[:, :-1], labels[:, 1:]))
        ems.append(exact_match(logits[:, :-1], labels[:, 1:]))

    return {
        "loss": float(np.mean(losses)),
        "token_acc": float(np.mean(accs)),
        "exact_match": float(np.mean(ems)),
    }


def build_dataset(qa_pairs, tokenizer, max_len, perf: PerfTracker):
    dataset = []

    for idx, s in enumerate(qa_pairs):
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


def run_experiment(backend: str):
    perf = PerfTracker(backend)
    print(f"[{backend}] === run_experiment:start ===", flush=True)

    loss_backend = backend
    ignore_index = 0
    lr_decoder = 1e-3
    lr_query = 1e-3

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

    with perf.track("tokenizer.build"):
        tokenizer = build_tokenizer("bert-base-uncased")

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

    total_train_loss = 0.0

    print(f"[{backend}] train:start epochs=2", flush=True)
    for epoch in range(2):
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

            print(
                f"[{backend}] epoch={epoch} step={step} "
                f"loss={float(loss):.6f} "
                f"query_grad_norm={float(np.linalg.norm(grad_query)):.6f} "
                f"sample_time={sample_dt:.6f}s",
                flush=True,
            )

        perf.add("train.epoch.total", time.perf_counter() - epoch_t0)

        with perf.track("dev.evaluate.total"):
            metrics = evaluate(
                pipeline=pipeline,
                bank=bank,
                data=dev_data,
                loss_backend=loss_backend,
                ignore_index=ignore_index,
                perf=perf,
            )

        print(
            f"[{backend}] epoch={epoch}:done "
            f"loss={metrics['loss']:.4f} "
            f"token_acc={metrics['token_acc']:.4f} "
            f"exact_match={metrics['exact_match']:.4f}",
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


def main():
    for backend in ["cython", "numpy"]:
        print("\n==============================")
        print(f"RUNNING BACKEND = {backend}")
        print("==============================\n")
        run_experiment(backend)


if __name__ == "__main__":
    main()