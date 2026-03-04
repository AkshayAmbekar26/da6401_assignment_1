"""
Finite-difference gradient checks for NeuralNetwork backward pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from ann.neural_network import NeuralNetwork


@dataclass
class Args:
    dataset: str = "mnist"
    epochs: int = 1
    batch_size: int = 4
    loss: str = "cross_entropy"
    optimizer: str = "sgd"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_layers: int = 2
    hidden_size: tuple[int, ...] = (8, 8)
    activation: str = "tanh"
    weight_init: str = "xavier"
    wandb_project: str = "da6401_assignment_1"
    wandb_entity: str | None = None
    wandb_mode: str = "disabled"
    seed: int = 7
    model_path: str = "src/best_model.npy"
    config_path: str = "src/best_config.json"


def relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(1e-12, abs(a) + abs(b))


def compute_loss(model: NeuralNetwork, X: np.ndarray, y_onehot: np.ndarray) -> float:
    logits = model.forward(X)
    model.backward(y_onehot, logits)
    return float(model.last_loss)


def run_gradient_check(epsilon: float, checks_per_matrix: int) -> float:
    args = Args()
    model = NeuralNetwork(args)
    rng = np.random.default_rng(args.seed)

    X = rng.standard_normal((4, 28 * 28)).astype(np.float64)
    y = rng.integers(0, 10, size=(4,))
    y_onehot = np.zeros((4, 10), dtype=np.float64)
    y_onehot[np.arange(4), y] = 1.0

    logits = model.forward(X)
    grad_W, grad_b = model.backward(y_onehot, logits)

    max_err = 0.0
    for rev_idx, layer in enumerate(model.layers[::-1]):
        gw = grad_W[rev_idx]
        gb = grad_b[rev_idx]

        w_indices = list(zip(*np.unravel_index(
            rng.choice(layer.W.size, size=min(checks_per_matrix, layer.W.size), replace=False),
            layer.W.shape
        )))
        b_indices = list(zip(*np.unravel_index(
            rng.choice(layer.b.size, size=min(checks_per_matrix, layer.b.size), replace=False),
            layer.b.shape
        )))

        for i, j in w_indices:
            old = layer.W[i, j]
            layer.W[i, j] = old + epsilon
            loss_plus = compute_loss(model, X, y_onehot)
            layer.W[i, j] = old - epsilon
            loss_minus = compute_loss(model, X, y_onehot)
            layer.W[i, j] = old
            num_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
            err = relative_error(num_grad, float(gw[i, j]))
            max_err = max(max_err, err)

        for i, j in b_indices:
            old = layer.b[i, j]
            layer.b[i, j] = old + epsilon
            loss_plus = compute_loss(model, X, y_onehot)
            layer.b[i, j] = old - epsilon
            loss_minus = compute_loss(model, X, y_onehot)
            layer.b[i, j] = old
            num_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
            err = relative_error(num_grad, float(gb[i, j]))
            max_err = max(max_err, err)

    return max_err


def main():
    parser = argparse.ArgumentParser(description="Gradient check for NumPy MLP")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--checks_per_matrix", type=int, default=20)
    args = parser.parse_args()

    max_err = run_gradient_check(args.epsilon, args.checks_per_matrix)
    print(f"Max relative error: {max_err:.12e}")
    if max_err < 1e-7:
        print("PASS: Numerical and analytical gradients are consistent.")
    else:
        print("WARN: Relative error above target threshold 1e-7.")


if __name__ == "__main__":
    main()
