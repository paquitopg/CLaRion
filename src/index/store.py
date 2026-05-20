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
    __slots__ = ("index_path", "meta_path", "_bank", "_meta")

    def __init__(self, index_path: str | os.PathLike, meta_path: str | os.PathLike):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self._bank: Optional[np.ndarray] = None
        self._meta: Optional[dict] = None

    def save(
        self,
        bank: np.ndarray,
        config: ModelConfig,
        extra: Optional[dict] = None,
    ) -> None:
        bank = np.asarray(bank, dtype=np.float32, order="C")

        if bank.ndim not in (2, 3):
            raise ValueError(f"bank must be 2D or 3D, got shape {bank.shape}")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(self.index_path, bank, allow_pickle=False)

        meta = {
            "shape": list(bank.shape),
            "n_docs": int(bank.shape[0]),
            "dtype": str(bank.dtype),
            "config": asdict(config),
            "config_hash": _hash_config(config),
        }

        if bank.ndim == 2:
            meta["embedding_dim"] = int(bank.shape[1])
        else:
            meta["n_memory_tokens"] = int(bank.shape[1])
            meta["hidden_dim"] = int(bank.shape[2])
            meta["embedding_dim"] = int(bank.shape[1] * bank.shape[2])

        if extra:
            meta.update(extra)

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Saved index: shape=%s -> %s",
            bank.shape,
            self.index_path,
        )

    def load(self, mmap: bool = True) -> tuple[np.ndarray, dict]:
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)

        bank = np.load(
            self.index_path,
            mmap_mode="r" if mmap else None,
        )
        bank = np.asarray(bank, dtype=np.float32, order="C")

        if not self.meta_path.exists():
            raise FileNotFoundError(self.meta_path)

        with self.meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        self._bank = bank
        self._meta = meta
        return bank, meta

    @property
    def bank(self) -> np.ndarray:
        if self._bank is None:
            self.load()
        return self._bank

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self.load()
        return self._meta


def _hash_config(config: ModelConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]