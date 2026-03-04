"""
Loss functions that operate directly on logits.
"""

from __future__ import annotations

import numpy as np

from ann.activations import softmax


def cross_entropy_from_logits(logits: np.ndarray, y_onehot: np.ndarray) -> tuple[float, np.ndarray]:
    probs = softmax(logits)
    eps = 1e-12
    log_probs = np.log(np.clip(probs, eps, 1.0))
    batch_size = logits.shape[0]
    loss = -np.sum(y_onehot * log_probs) / batch_size
    dlogits = (probs - y_onehot) / batch_size
    return float(loss), dlogits


def mse_from_logits(logits: np.ndarray, y_onehot: np.ndarray) -> tuple[float, np.ndarray]:
    batch_size = logits.shape[0]
    diff = logits - y_onehot
    loss = np.mean(np.sum(diff * diff, axis=1))
    dlogits = 2.0 * diff / batch_size
    return float(loss), dlogits


LOSSES = {
    "cross_entropy": cross_entropy_from_logits,
    "mean_squared_error": mse_from_logits,
    "mse": mse_from_logits,
}
