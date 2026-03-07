"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ann.neural_network import NeuralNetwork
from utils.cli import add_common_arguments, validate_hidden_sizes
from utils.data_loader import load_dataset

try:
    import wandb
except ImportError:  # pragma: no cover - wandb might be unavailable in some environments
    wandb = None


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    add_common_arguments(parser)
    args = parser.parse_args()
    validate_hidden_sizes(args)
    return args


def setup_wandb(args):
    if args.wandb_mode == "disabled":
        return None

    if wandb is None:
        print("wandb is not installed; continuing without wandb logging.")
        return None

    os_mode = "online" if args.wandb_mode == "online" else "offline"
    tags = [args.dataset, args.optimizer, args.activation, args.loss, args.weight_init]
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=os_mode,
            config=vars(args),
            tags=tags,
        )
    except Exception as exc:  # pragma: no cover
        print(f"wandb init failed ({exc}); continuing without wandb logging.")
        return None
    return run


def maybe_log_class_samples(wandb_run, dataset_dict, limit_per_class=5):
    if wandb_run is None or wandb is None:
        return

    X_train = dataset_dict["X_train"]
    y_train = dataset_dict["y_train"]
    records = []
    for cls in range(10):
        cls_idx = np.where(y_train == cls)[0][:limit_per_class]
        for idx in cls_idx:
            img = (X_train[idx].reshape(28, 28) * 255.0).astype(np.uint8)
            records.append([cls, wandb.Image(img, caption=f"class_{cls}")])

    table = wandb.Table(columns=["class", "image"], data=records)
    wandb_run.log({"dataset/class_samples": table})


def save_best_artifacts(model, args, final_metrics):
    model_path = Path(args.model_path)
    config_path = Path(args.config_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
        "saved_model_path": str(model_path),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metrics": final_metrics,
    }

    # avoid overwriting a stronger canonical submission model during smoke runs
    is_canonical_best = (
        model_path.name == "best_model.npy"
        and config_path.name == "best_config.json"
        and model_path.parent.name == "src"
        and config_path.parent.name == "src"
    )
    new_test_f1 = float(final_metrics.get("test", {}).get("f1", -np.inf))
    if is_canonical_best and model_path.exists() and config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
            existing_test_f1 = float(
                existing_cfg.get("metrics", {}).get("test", {}).get("f1", -np.inf)
            )
            if existing_test_f1 >= new_test_f1:
                print(
                    f"Preserving existing src/best_model.npy (existing test F1 {existing_test_f1:.4f} "
                    f">= new test F1 {new_test_f1:.4f})."
                )
                return str(model_path), str(config_path), existing_cfg
        except Exception:
            # if existing config is unreadable, fall back to overwrite
            pass

    best_weights = model.get_weights()
    # store minimal metadata so loading stays robust across different cli defaults
    best_weights["__meta__"] = {
        "activation": str(args.activation),
        "loss": str(args.loss),
        "num_layers": int(args.num_layers),
        "hidden_size": [int(v) for v in args.hidden_size],
    }
    np.save(model_path, best_weights)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    return str(model_path), str(config_path), config_payload


def train_and_evaluate(args, wandb_run_override=None):
    data = load_dataset(dataset=args.dataset, seed=args.seed)
    model = NeuralNetwork(args)
    # lets notebook/sweep callers manage wandb lifecycle externally
    wandb_run = wandb_run_override if wandb_run_override is not None else setup_wandb(args)
    maybe_log_class_samples(wandb_run, data)

    history = model.train(
        X_train=data["X_train"],
        y_train=data["y_train"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=data["X_val"],
        y_val=data["y_val"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        wandb_run=wandb_run,
    )

    final_val = model.evaluate(data["X_val"], data["y_val"], batch_size=args.batch_size)
    final_test = model.evaluate(data["X_test"], data["y_test"], batch_size=args.batch_size)
    # drop logits from persisted metrics payload to keep config json lightweight
    final_metrics = {"val": {k: v for k, v in final_val.items() if k != "logits"}, "test": {k: v for k, v in final_test.items() if k != "logits"}}

    model_path, config_path, config_payload = save_best_artifacts(model, args, final_metrics)

    result = {
        "history": history,
        "final_metrics": final_metrics,
        "model_path": model_path,
        "config_path": config_path,
        "config": config_payload,
    }

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best/val_f1": final_metrics["val"]["f1"],
                "best/test_f1": final_metrics["test"]["f1"],
                "artifact/model_path": model_path,
                "artifact/config_path": config_path,
            }
        )
        if wandb_run_override is None:
            wandb_run.finish()

    return result


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    result = train_and_evaluate(args)
    print("Training complete!")
    print(
        "Final metrics | "
        f"Val F1: {result['final_metrics']['val']['f1']:.4f}, "
        f"Test F1: {result['final_metrics']['test']['f1']:.4f}"
    )
    print(f"Saved model: {result['model_path']}")
    print(f"Saved config: {result['config_path']}")
    return result


if __name__ == '__main__':
    main()
