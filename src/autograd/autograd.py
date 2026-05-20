import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None

        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        self.grad = np.ones_like(self.data)
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for v in reversed(topo):
            v._backward()

    # ops

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T if self.grad is not None else out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad if other.grad is not None else self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (self.data > 0) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Adam:
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue

            m = self.m[id(p)]
            v = self.v[id(p)]

            m[:] = 0.9 * m + 0.1 * p.grad
            v[:] = 0.999 * v + 0.001 * (p.grad ** 2)

            m_hat = m / (1 - 0.9 ** self.t)
            v_hat = v / (1 - 0.999 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            p.grad = None