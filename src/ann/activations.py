"""
Activation functions and derivatives for hidden layers.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
}
