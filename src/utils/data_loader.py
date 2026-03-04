"""
Dataset loading and preprocessing utilities.
"""

from __future__ import annotations

import numpy as np
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split


def load_dataset(dataset: str = "mnist", seed: int = 42, val_size: float = 0.1):
    key = dataset.lower()
    if key == "mnist":
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    elif key == "fashion_mnist":
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    X_train_full = preprocess_images(X_train_full)
    X_test = preprocess_images(X_test)
    y_train_full = y_train_full.astype(np.int64)
    y_test = y_test.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def preprocess_images(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float64) / 255.0
    return images.reshape(images.shape[0], -1)


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = y.astype(int)
    encoded = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def batch_iterator(X, y, batch_size: int, rng: np.random.Generator | None = None, shuffle: bool = True):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.permutation(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]
