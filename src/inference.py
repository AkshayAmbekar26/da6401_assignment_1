"""
Inference Script
Evaluate trained models on test sets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ann.neural_network import NeuralNetwork
from utils.cli import add_common_arguments, validate_hidden_sizes
from utils.data_loader import load_dataset

def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    add_common_arguments(parser)
    args = parser.parse_args()
    validate_hidden_sizes(args)
    return args


def load_model(model_path):
    """
    Load trained model from disk.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError("Model file must contain a pickled dictionary of weights.")
    return data


def evaluate_model(model, X_test, y_test, batch_size): 
    """
    Evaluate model on test data.
    """
    metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
    return {
        "logits": metrics["logits"],
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def main():
    """
    Main inference function.
    """
    args = parse_arguments()
    data = load_dataset(dataset=args.dataset, seed=args.seed)
    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)
    metrics = evaluate_model(model, data["X_test"], data["y_test"], batch_size=args.batch_size)

    print("Evaluation complete!")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Loss:      {metrics['loss']:.4f}")
    return metrics


if __name__ == '__main__':
    main()
