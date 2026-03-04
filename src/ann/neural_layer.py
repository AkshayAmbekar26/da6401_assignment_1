"""
Fully-connected layer implementation.
"""

from __future__ import annotations

import numpy as np


class DenseLayer:
    """
    Dense layer with cached inputs and explicit parameter gradients.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_init: str = "xavier",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight_init = weight_init
        self.rng = rng if rng is not None else np.random.default_rng()

        self.W, self.b = self._initialize_params(weight_init)
        self.X_cache: np.ndarray | None = None
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def _initialize_params(self, weight_init: str) -> tuple[np.ndarray, np.ndarray]:
        if weight_init == "random":
            W = self.rng.standard_normal((self.input_dim, self.output_dim)) * 0.01
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (self.input_dim + self.output_dim))
            W = self.rng.standard_normal((self.input_dim, self.output_dim)) * std
        elif weight_init == "zeros":
            W = np.zeros((self.input_dim, self.output_dim), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported weight_init: {weight_init}")

        b = np.zeros((1, self.output_dim), dtype=np.float64)
        return W.astype(np.float64), b

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_cache = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        if self.X_cache is None:
            raise RuntimeError("Cannot run backward before forward on DenseLayer.")

        self.grad_W = self.X_cache.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ @ self.W.T
        return dA_prev
