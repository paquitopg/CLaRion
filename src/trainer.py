import time
import numpy as np

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .models.config import (
    ModelConfig,
    IndexConfig,
    DecoderConfig,
    TopKConfig,
)

from .models.encoder import build_encoder
from .models.clara_decoder import build_decoder
from .models.topk import build_topk
from .models.pipeline import ClaraPipeline
from .models.loss import cross_entropy_with_grad

from .index.builder import IndexBuilder
from .data.fallback_sample import (
    BUNDLED_DOCS,
    SAMPLE_QA_PAIRS,
)


def time_block():
    return time.perf_counter()


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


def build_qa_dataset(
    qa_pairs,
    tokenizer,
    max_len: int,
):
    dataset = []

    for sample in qa_pairs:

        question = sample["question"]
        answer = sample["answer"]

        prompt = f"question: {question} answer:"

        x = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="np",
        )

        y = tokenizer(
            answer,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="np",
        )

        dataset.append({
            "input_ids": x["input_ids"].astype(np.int32),
            "labels": y["input_ids"].astype(np.int32),
            "supporting_docs": sample["supporting_doc_indices"],
        })

    return dataset


def token_accuracy(logits, labels):

    preds = np.argmax(logits, axis=-1)

    mask = labels != 0

    correct = ((preds == labels) & mask).sum()
    total = np.maximum(mask.sum(), 1)

    return float(correct / total)


def exact_match(logits, labels):

    preds = np.argmax(logits, axis=-1)

    return float(
        np.all(preds == labels, axis=-1).mean()
    )


def evaluate(
    pipeline,
    bank,
    data,
):

    losses = []
    accs = []
    ems = []

    for input_ids, labels in data:

        logits, _ = pipeline.forward(
            input_ids=input_ids,
            bank=bank,
        )

        loss, _ = cross_entropy_with_grad(
            logits,
            labels,
            ignore_index=0,
        )

        losses.append(loss)

        accs.append(
            token_accuracy(
                logits[:, :-1],
                labels[:, 1:],
            )
        )

        ems.append(
            exact_match(
                logits[:, :-1],
                labels[:, 1:],
            )
        )

    return {
        "loss": float(np.mean(losses)),
        "token_acc": float(np.mean(accs)),
        "exact_match": float(np.mean(ems)),
    }


def run_experiment(backend: str):
    print(f"[{backend}] === run_experiment:start ===", flush=True)

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
    print(f"[{backend}] cfg built", flush=True)

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
    print(f"[{backend}] decoder_cfg built", flush=True)

    index_cfg = IndexConfig(
        index_path="./artifacts/index.npy",
        meta_path="./artifacts/index_meta.json",
        batch_size=8,
    )
    print(f"[{backend}] index_cfg built", flush=True)

    topk_cfg = TopKConfig(
        k=4,
        temperature=1.0,
    )
    print(f"[{backend}] topk_cfg built", flush=True)

    print(f"[{backend}] tokenizer:start", flush=True)
    tokenizer = build_tokenizer("bert-base-uncased")
    print(f"[{backend}] tokenizer:done", flush=True)

    dataset = []
    print(f"[{backend}] dataset_build:start n_samples={len(SAMPLE_QA_PAIRS)}", flush=True)

    for idx, s in enumerate(SAMPLE_QA_PAIRS):
        print(f"[{backend}] dataset_build:sample={idx}:tokenize_prompt:start", flush=True)
        x = tokenizer(
            f"question: {s['question']} answer:",
            truncation=True,
            max_length=cfg.max_seq_len,
            padding="max_length",
            return_tensors="np",
        )
        print(f"[{backend}] dataset_build:sample={idx}:tokenize_prompt:done", flush=True)

        print(f"[{backend}] dataset_build:sample={idx}:tokenize_answer:start", flush=True)
        y = tokenizer(
            s["answer"],
            truncation=True,
            max_length=cfg.max_seq_len,
            padding="max_length",
            return_tensors="np",
        )
        print(f"[{backend}] dataset_build:sample={idx}:tokenize_answer:done", flush=True)

        dataset.append((
            x["input_ids"].astype(np.int32),
            y["input_ids"].astype(np.int32),
        ))

        print(
            f"[{backend}] dataset_build:sample={idx}:done "
            f"input_shape={dataset[-1][0].shape} labels_shape={dataset[-1][1].shape}",
            flush=True,
        )

    print(f"[{backend}] dataset_build:done total={len(dataset)}", flush=True)

    print(f"[{backend}] split:start", flush=True)
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
    print(
        f"[{backend}] split:done "
        f"train={len(train_data)} dev={len(dev_data)} test={len(test_data)}",
        flush=True,
    )

    print(f"[{backend}] builder:init:start", flush=True)
    builder = IndexBuilder(
        model_config=cfg,
        index_config=index_cfg,
        tokenizer_name="bert-base-uncased",
    )
    print(f"[{backend}] builder:init:done", flush=True)

    print(
        f"[{backend}] builder:build:start "
        f"docs={len(BUNDLED_DOCS)} parallel={(backend == 'cython')}",
        flush=True,
    )
    t0 = time.perf_counter()

    bank, report = builder.build(
        docs=BUNDLED_DOCS,
        backend=backend,
        parallel=(backend == "cython"),
        save=False,
    )

    t1 = time.perf_counter()
    print(f"[{backend}] builder:build:done elapsed={t1 - t0:.4f}s", flush=True)

    if hasattr(bank, "shape"):
        print(f"[{backend}] bank.shape={bank.shape}", flush=True)
    else:
        print(f"[{backend}] bank.type={type(bank)}", flush=True)

    print(f"[{backend}] encoder_build:start", flush=True)
    encoder = build_encoder(cfg, backend=backend)
    print(f"[{backend}] encoder_build:done type={type(encoder).__name__}", flush=True)

    print(f"[{backend}] decoder_build:start", flush=True)
    decoder = build_decoder(
        decoder_cfg,
        backend=backend,
    )
    print(f"[{backend}] decoder_build:done type={type(decoder).__name__}", flush=True)

    print(f"[{backend}] topk_build:start", flush=True)
    topk = build_topk(
        topk_cfg,
        backend=backend,
    )
    print(f"[{backend}] topk_build:done type={type(topk).__name__}", flush=True)

    print(f"[{backend}] pipeline:init:start", flush=True)
    pipeline = ClaraPipeline(
        encoder=encoder,
        decoder=decoder,
        topk=topk,
    )
    print(f"[{backend}] pipeline:init:done", flush=True)

    total_train_time = 0.0

    print(f"[{backend}] train:start epochs=2", flush=True)
    for epoch in range(2):
        print(f"[{backend}] epoch={epoch}:start", flush=True)

        epoch_start = time.perf_counter()
        total_loss = 0.0

        for step, (input_ids, labels) in enumerate(train_data):
            print(
                f"[{backend}] epoch={epoch} step={step}:sample:start "
                f"input_shape={input_ids.shape} labels_shape={labels.shape}",
                flush=True,
            )

            sample_t0 = time.perf_counter()

            print(f"[{backend}] epoch={epoch} step={step}:forward:start", flush=True)
            logits, aux = pipeline.forward(
                input_ids=input_ids,
                bank=bank,
            )
            print(
                f"[{backend}] epoch={epoch} step={step}:forward:done "
                f"logits_shape={logits.shape} aux_type={type(aux).__name__}",
                flush=True,
            )

            print(f"[{backend}] epoch={epoch} step={step}:loss:start", flush=True)
            loss, grad_logits = cross_entropy_with_grad(
                logits,
                labels,
                ignore_index=0,
            )
            print(
                f"[{backend}] epoch={epoch} step={step}:loss:done "
                f"loss={float(loss):.6f} grad_shape={grad_logits.shape}",
                flush=True,
            )

            print(f"[{backend}] epoch={epoch} step={step}:backward:start", flush=True)
            decoder.backward(
                grad_logits,
                lr=1e-3,
            )
            print(f"[{backend}] epoch={epoch} step={step}:backward:done", flush=True)

            sample_t1 = time.perf_counter()
            total_loss += loss
            total_train_time += (sample_t1 - sample_t0)

            print(
                f"[{backend}] epoch={epoch} step={step}:sample:done "
                f"elapsed={sample_t1 - sample_t0:.4f}s",
                flush=True,
            )

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"[{backend}] epoch={epoch}:done "
            f"loss={total_loss:.4f} epoch_time={epoch_time:.4f}s",
            flush=True,
        )

        print(f"[{backend}] epoch={epoch}:eval:start", flush=True)
        metrics = evaluate(
            pipeline=pipeline,
            bank=bank,
            data=dev_data,
        )
        print(
            f"[{backend}] epoch={epoch}:eval:done "
            f"loss={metrics['loss']:.4f} "
            f"token_acc={metrics['token_acc']:.4f} "
            f"exact_match={metrics['exact_match']:.4f}",
            flush=True,
        )

    print(f"[{backend}] TOTAL TRAIN TIME = {total_train_time:.4f}s", flush=True)
    print(f"[{backend}] === run_experiment:end ===", flush=True)

def main():

    for backend in ["cython", "numpy"]:

        print("\n==============================")
        print(f"RUNNING BACKEND = {backend}")
        print("==============================\n")

        run_experiment(backend)


if __name__ == "__main__":
    main()