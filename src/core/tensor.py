from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from ..models.config import TensorEngineConfig


logger = logging.getLogger("clarion.tensor_engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TensorEngineBackend(ABC):
    """
    Abstract backend for tensor/autograd helper kernels.

    This backend is used by the Tensor class to dispatch numerical kernels.
    The graph/autograd logic stays in Python; hot kernels can be NumPy or Cython.
    """

    kind = "abstract"

    def __init__(self, config: TensorEngineConfig):
        self.config = config

    @abstractmethod
    def sum_to_shape(self, grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def relu_forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def relu_backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def softmax_forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def softmax_backward(
        self,
        probs: np.ndarray,
        grad_out: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        raise NotImplementedError


class TensorEngineNumpy(TensorEngineBackend):
    """
    Pure NumPy backend.

    Reference implementation: simple, explicit, easy to test.
    """

    kind = "numpy"

    def sum_to_shape(self, grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        grad = np.ascontiguousarray(grad, dtype=np.float32)

        if grad.shape == shape:
            return grad

        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
            if sdim == 1 and gdim != 1:
                grad = grad.sum(axis=i, keepdims=True)

        return np.ascontiguousarray(grad.reshape(shape), dtype=np.float32)

    def relu_forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        return np.maximum(x, 0.0).astype(np.float32, copy=False)

    def relu_backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        grad_out = np.ascontiguousarray(grad_out, dtype=np.float32)
        return ((x > 0).astype(np.float32) * grad_out).astype(np.float32, copy=False)

    def softmax_forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return (e / (np.sum(e, axis=axis, keepdims=True) + self.config.eps)).astype(np.float32, copy=False)

    def softmax_backward(
        self,
        probs: np.ndarray,
        grad_out: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        probs = np.ascontiguousarray(probs, dtype=np.float32)
        grad_out = np.ascontiguousarray(grad_out, dtype=np.float32)
        dot = np.sum(grad_out * probs, axis=axis, keepdims=True)
        return (probs * (grad_out - dot)).astype(np.float32, copy=False)


class TensorEngineCython(TensorEngineBackend):
    """
    Hybrid backend: Cython when available, NumPy fallback otherwise.

    The API stays identical to TensorEngineNumpy.
    """

    kind = "cython"

    def __init__(self, config: TensorEngineConfig):
        super().__init__(config)

        self._available = False
        self._ext = None
        self._numpy = TensorEngineNumpy(config)

        try:
            from src.parallel import cython_tensor_engine
            self._ext = cython_tensor_engine
            self._available = True
            logger.info("Cython tensor engine loaded successfully.")
        except Exception as e:
            logger.warning("Cython tensor engine unavailable: %s", e)
            self._ext = None

    @property
    def is_available(self) -> bool:
        return self._available

    def sum_to_shape(self, grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        # This op is mostly reduction/broadcast logic; NumPy is fine here.
        return self._numpy.sum_to_shape(grad, shape)

    def relu_forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)

        if not self._available:
            return self._numpy.relu_forward(x)

        return self._ext.relu_forward(x)

    def relu_backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        grad_out = np.ascontiguousarray(grad_out, dtype=np.float32)

        if not self._available:
            return self._numpy.relu_backward(x, grad_out)

        return self._ext.relu_backward(x, grad_out)

    def softmax_forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)

        if axis != -1:
            return self._numpy.softmax_forward(x, axis=axis)

        if not self._available:
            return self._numpy.softmax_forward(x, axis=axis)

        return self._ext.softmax_lastdim_forward(
            x,
            self.config.eps,
            self.config.num_threads,
        )

    def softmax_backward(
        self,
        probs: np.ndarray,
        grad_out: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        probs = np.ascontiguousarray(probs, dtype=np.float32)
        grad_out = np.ascontiguousarray(grad_out, dtype=np.float32)

        if axis != -1:
            return self._numpy.softmax_backward(probs, grad_out, axis=axis)

        if not self._available:
            return self._numpy.softmax_backward(probs, grad_out, axis=axis)

        return self._ext.softmax_lastdim_backward(
            probs,
            grad_out,
            self.config.num_threads,
        )


def build_tensor_engine(
    config: TensorEngineConfig,
    backend: str = "numpy",
) -> TensorEngineBackend:
    """
    Factory for tensor engines.
    """
    if backend == "numpy":
        return TensorEngineNumpy(config)

    if backend == "cython":
        return TensorEngineCython(config)

    if backend == "auto":
        if config.use_cython:
            return TensorEngineCython(config)
        return TensorEngineNumpy(config)

    raise ValueError(f"Unknown backend: {backend}")




class Tensor:
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        engine: TensorEngineBackend | None = None,
        _children=(),
        _op: str = "",
    ):
        if engine is None:
            raise ValueError("Tensor requires an engine backend")

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self.engine = engine

        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def backend(self) -> str:
        return self.engine.kind

    @classmethod
    def parameter(cls, data, engine: TensorEngineBackend) -> "Tensor":
        return cls(
            data=data,
            requires_grad=True,
            engine=engine,
            _children=(),
            _op="parameter",
        )

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.data.shape}, "
            f"requires_grad={self.requires_grad}, "
            f"backend={self.backend})"
        )