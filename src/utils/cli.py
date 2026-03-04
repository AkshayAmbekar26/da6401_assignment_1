"""
Shared CLI argument parsing for training and inference.
"""

from __future__ import annotations

import argparse


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-d", "-dataset", "--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "-epochs", "--epochs", type=int, default=20)
    parser.add_argument("-b", "-batch_size", "--batch_size", type=int, default=64)
    parser.add_argument(
        "-l",
        "-loss",
        "--loss",
        default="cross_entropy",
        choices=["cross_entropy", "mean_squared_error", "mse"],
    )
    parser.add_argument("-o", "-optimizer", "--optimizer", default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr", "-learning_rate", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("-nhl", "--num_layers", type=int, default=4)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128, 128, 128, 128])
    parser.add_argument("-a", "--activation", default="relu", choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init", default="xavier", choices=["random", "xavier", "zeros"])
    parser.add_argument("-w_p", "--wandb_project", default="da6401_assignment_1")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", default="src/best_model.npy")
    parser.add_argument("--config_path", default="src/best_config.json")
    return parser


def validate_hidden_sizes(args) -> None:
    if args.num_layers < 1:
        raise ValueError("--num_layers must be >= 1")

    if len(args.hidden_size) == 1 and args.num_layers > 1:
        args.hidden_size = args.hidden_size * args.num_layers
        return

    if len(args.hidden_size) == args.num_layers:
        return

    raise ValueError(
        "--hidden_size must contain either one value (replicated) or exactly --num_layers values."
    )
