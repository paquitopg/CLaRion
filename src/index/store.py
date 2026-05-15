"""
On-disk store for the document embedding bank.

Layout
------
- index_path   :  flat (N, D) float32 array (numpy .npy)
- meta_path    :  JSON sidecar with N, D, dtype, source-doc range, encoder
                  config hash (so we can detect mismatched indices later).

The whole bank lives in RAM at retrieval time; for small toy corpora (1k..100k
docs * a few KB each) that's fine. We keep the on-disk format simple — no
HNSW / IVF / quantization — because the project is a CPU-parallelization
study, not a billion-scale ANN benchmark.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..models.config import ModelConfig

logger = logging.getLogger("clarion.index.store")


class IndexStore:
    """Tiny memory-mapped index store."""

    __slots__ = ("index_path", "meta_path", "_bank", "_meta")

    def __init__(self, index_path: str | os.PathLike, meta_path: str | os.PathLike):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self._bank: Optional[np.ndarray] = None
        self._meta: Optional[dict] = None

    # --------------------------- write ---------------------------- #
    def save(self, bank: np.ndarray, config: ModelConfig, extra: Optional[dict] = None) -> None:
        assert bank.dtype == np.float32
        assert bank.ndim == 2

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(self.index_path, bank, allow_pickle=False)

        meta = {
            "n_docs": int(bank.shape[0]),
            "embedding_dim": int(bank.shape[1]),
            "dtype": str(bank.dtype),
            "config": asdict(config),
            "config_hash": _hash_config(config),
        }
        if extra is not None:
            meta.update(extra)
        with self.meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Wrote %d-doc index of dim %d to %s",
                    bank.shape[0], bank.shape[1], self.index_path)

    # --------------------------- read ----------------------------- #
    def load(self, mmap: bool = True) -> tuple[np.ndarray, dict]:
        bank = np.load(self.index_path, mmap_mode="r" if mmap else None)
        bank = np.ascontiguousarray(bank, dtype=np.float32)
        with self.meta_path.open("r") as f:
            meta = json.load(f)
        self._bank = bank
        self._meta = meta
        return bank, meta

    @property
    def bank(self) -> np.ndarray:
        if self._bank is None:
            self.load()
        return self._bank  # type: ignore[return-value]

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self.load()
        return self._meta  # type: ignore[return-value]


def _hash_config(config: ModelConfig) -> str:
    """Stable hash of the encoder config — used to spot stale indices."""
    payload = json.dumps(asdict(config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
