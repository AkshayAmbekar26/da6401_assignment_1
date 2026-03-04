"""
Optimization algorithms for updating DenseLayer parameters.
"""

from __future__ import annotations

import numpy as np


class BaseOptimizer:
    def __init__(self, learning_rate: float, weight_decay: float = 0.0) -> None:
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)

    def _apply_weight_decay(self, layer) -> np.ndarray:
        if self.weight_decay <= 0.0:
            return layer.grad_W
        return layer.grad_W + self.weight_decay * layer.W

    def step(self, layers) -> None:
        raise NotImplementedError


class SGDOptimizer(BaseOptimizer):
    def step(self, layers) -> None:
        for layer in layers:
            dW = self._apply_weight_decay(layer)
            db = layer.grad_b
            layer.W -= self.learning_rate * dW
            layer.b -= self.learning_rate * db


class MomentumOptimizer(BaseOptimizer):
    def __init__(self, learning_rate: float, weight_decay: float = 0.0, beta: float = 0.9) -> None:
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def step(self, layers) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.vW:
                self.vW[idx] = np.zeros_like(layer.W)
                self.vb[idx] = np.zeros_like(layer.b)

            dW = self._apply_weight_decay(layer)
            db = layer.grad_b

            self.vW[idx] = self.beta * self.vW[idx] + (1.0 - self.beta) * dW
            self.vb[idx] = self.beta * self.vb[idx] + (1.0 - self.beta) * db

            layer.W -= self.learning_rate * self.vW[idx]
            layer.b -= self.learning_rate * self.vb[idx]


class NAGOptimizer(BaseOptimizer):
    def __init__(self, learning_rate: float, weight_decay: float = 0.0, beta: float = 0.9) -> None:
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def step(self, layers) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.vW:
                self.vW[idx] = np.zeros_like(layer.W)
                self.vb[idx] = np.zeros_like(layer.b)

            dW = self._apply_weight_decay(layer)
            db = layer.grad_b

            vW_prev = self.vW[idx].copy()
            vb_prev = self.vb[idx].copy()

            self.vW[idx] = self.beta * self.vW[idx] - self.learning_rate * dW
            self.vb[idx] = self.beta * self.vb[idx] - self.learning_rate * db

            layer.W += -self.beta * vW_prev + (1.0 + self.beta) * self.vW[idx]
            layer.b += -self.beta * vb_prev + (1.0 + self.beta) * self.vb[idx]


class RMSPropOptimizer(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        beta: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.sW = {}
        self.sb = {}

    def step(self, layers) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.sW:
                self.sW[idx] = np.zeros_like(layer.W)
                self.sb[idx] = np.zeros_like(layer.b)

            dW = self._apply_weight_decay(layer)
            db = layer.grad_b

            self.sW[idx] = self.beta * self.sW[idx] + (1.0 - self.beta) * (dW * dW)
            self.sb[idx] = self.beta * self.sb[idx] + (1.0 - self.beta) * (db * db)

            layer.W -= self.learning_rate * dW / (np.sqrt(self.sW[idx]) + self.epsilon)
            layer.b -= self.learning_rate * db / (np.sqrt(self.sb[idx]) + self.epsilon)


def get_optimizer(name: str, learning_rate: float, weight_decay: float = 0.0):
    key = name.lower()
    if key == "sgd":
        return SGDOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if key == "momentum":
        return MomentumOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if key == "nag":
        return NAGOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if key == "rmsprop":
        return RMSPropOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")
