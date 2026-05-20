from __future__ import annotations

import logging
import numpy as np

from .core.tensor import Tensor
from ..models.config import OptimizerConfig


logger = logging.getLogger("clarion.optimizer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class OptimizerBackend:
    def __init__(self, config: OptimizerConfig):
        self.config = config

    def step(self, params: list[Tensor]) -> None:
        raise NotImplementedError


class AdamNumpy(OptimizerBackend):
    def __init__(self, params: list[Tensor], config: OptimizerConfig):
        super().__init__(config)

        self.params = list(params)
        self.t = 0
        self.beta1, self.beta2 = config.betas

        self.m = {id(p): np.zeros_like(p.data, dtype=np.float32) for p in self.params}
        self.v = {id(p): np.zeros_like(p.data, dtype=np.float32) for p in self.params}

    def step(self, params: list[Tensor] | None = None) -> None:
        if params is None:
            params = self.params

        self.t += 1
        t = self.t

        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.config.eps
        lr = self.config.lr
        wd = self.config.weight_decay

        for p in params:
            if not p.requires_grad or p.grad is None:
                continue

            grad = np.ascontiguousarray(p.grad, dtype=np.float32)

            if wd != 0.0:
                grad = grad + wd * p.data

            pid = id(p)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(p.data, dtype=np.float32)
                self.v[pid] = np.zeros_like(p.data, dtype=np.float32)

            m = self.m[pid]
            v = self.v[pid]

            m[:] = beta1 * m + (1.0 - beta1) * grad
            v[:] = beta2 * v + (1.0 - beta2) * (grad * grad)

            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)

            p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
            p.grad = None


class AdamCython(OptimizerBackend):
    def __init__(self, params: list[Tensor], config: OptimizerConfig):
        super().__init__(config)

        self._available = False
        self._ext = None

        try:
            from src.parallel import cython_optimizer
            self._ext = cython_optimizer
            self._available = True
        except Exception as e:
            logger.warning("Cython optimizer unavailable: %s", e)

        self._numpy = AdamNumpy(params, config)
        self.params = list(params)
        self._params_prepared = False

    def _prepare_contiguous_params(self) -> None:
        if self._params_prepared:
            return

        for p in self.params:
            p.data = np.ascontiguousarray(p.data, dtype=np.float32)
            if p.grad is not None:
                p.grad = np.ascontiguousarray(p.grad, dtype=np.float32)

        self._params_prepared = True

    def step(self, params: list[Tensor] | None = None) -> None:
        if params is None:
            params = self.params

        if not self._available:
            self._numpy.step(params)
            return

        self._prepare_contiguous_params()
        cfg = self.config

        try:
            from src.parallel.cython_optimizer import adam_step_numpy_interface

            self._numpy.t += 1
            t = self._numpy.t

            beta1, beta2 = cfg.betas
            eps = cfg.eps
            lr = cfg.lr
            wd = cfg.weight_decay

            adam_step_numpy_interface(
                params,
                self._numpy.m,
                self._numpy.v,
                beta1,
                beta2,
                eps,
                lr,
                wd,
                t,
                cfg.num_threads,
            )

            for p in params:
                if p.grad is not None:
                    p.grad = None

        except Exception as e:
            logger.warning("Cython adam_step failed: %s; falling back to NumPy.", e)
            self._numpy.step(params)