"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ann.activations import ACTIVATIONS
from ann.neural_layer import DenseLayer
from ann.objective_functions import LOSSES, cross_entropy_from_logits, mse_from_logits
from ann.optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.args = cli_args
        self.input_dim = 28 * 28
        self.output_dim = 10
        self.activation_name = str(getattr(cli_args, "activation", "relu")).lower()
        self.loss_name = str(getattr(cli_args, "loss", "cross_entropy")).lower()
        self.learning_rate = float(getattr(cli_args, "learning_rate", 1e-3))
        self.weight_decay = float(getattr(cli_args, "weight_decay", 0.0))
        self.optimizer_name = str(getattr(cli_args, "optimizer", "sgd")).lower()
        self.weight_init = str(getattr(cli_args, "weight_init", "xavier")).lower()

        hidden_size_arg = self._extract_hidden_sizes(cli_args)
        num_layers_arg = self._extract_num_layers(cli_args, hidden_size_arg)
        self.num_layers = int(num_layers_arg)
        self.hidden_sizes = self._resolve_hidden_sizes(hidden_size_arg, self.num_layers)
        self.seed = int(getattr(cli_args, "seed", 42))
        self.rng = np.random.default_rng(self.seed)

        if self.activation_name not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        if self.loss_name not in LOSSES:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

        dims = [self.input_dim, *self.hidden_sizes, self.output_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(
                DenseLayer(
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    weight_init=self.weight_init,
                    rng=self.rng,
                )
            )

        self.activation_fn, self.activation_derivative_fn = ACTIVATIONS[self.activation_name]
        self.optimizer = get_optimizer(
            self.optimizer_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.hidden_pre_activations = []
        self.hidden_activations = []
        self.last_loss = None
        self.grad_W = np.empty(0, dtype=object)
        self.grad_b = np.empty(0, dtype=object)

    @staticmethod
    def _extract_hidden_sizes(cli_args):
        hidden = getattr(cli_args, "hidden_size", None)
        if hidden is None:
            hidden = getattr(cli_args, "hidden_sizes", None)
        if hidden is None:
            return [128]

        if isinstance(hidden, np.ndarray):
            hidden = hidden.tolist()
        if isinstance(hidden, (int, float)):
            return [int(hidden)]
        if isinstance(hidden, tuple):
            hidden = list(hidden)
        if isinstance(hidden, list):
            return [int(v) for v in hidden]
        return [128]

    @staticmethod
    def _extract_num_layers(cli_args, hidden_sizes):
        for key in ("num_layers", "num_hidden_layers", "hidden_layers", "nhl"):
            value = getattr(cli_args, key, None)
            if value is not None:
                return int(value)
        if len(hidden_sizes) > 1:
            return len(hidden_sizes)
        return 1

    @staticmethod
    def _resolve_hidden_sizes(hidden_size_arg, num_layers):
        if isinstance(hidden_size_arg, (int, float)):
            hidden_sizes = [int(hidden_size_arg)]
        else:
            hidden_sizes = [int(v) for v in hidden_size_arg]
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if len(hidden_sizes) == 1 and num_layers > 1:
            return hidden_sizes * num_layers
        if len(hidden_sizes) == num_layers:
            return hidden_sizes
        raise ValueError(
            "hidden_size must be either one value (replicated) or exactly num_layers values."
        )

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        A = X
        self.hidden_pre_activations = []
        self.hidden_activations = []

        for layer in self.layers[:-1]:
            Z = layer.forward(A)
            self.hidden_pre_activations.append(Z)
            A = self.activation_fn(Z)
            self.hidden_activations.append(A)

        logits = self.layers[-1].forward(A)
        return logits

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        if self.loss_name == "cross_entropy":
            loss, dA = cross_entropy_from_logits(y_pred, y_true)
        elif self.loss_name in {"mean_squared_error", "mse"}:
            loss, dA = mse_from_logits(y_pred, y_true)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

        self.last_loss = loss

        dA = self.layers[-1].backward(dA)
        grad_W_list.append(self.layers[-1].grad_W.copy())
        grad_b_list.append(self.layers[-1].grad_b.copy())

        for i in range(len(self.layers) - 2, -1, -1):
            dZ = dA * self.activation_derivative_fn(self.hidden_pre_activations[i])
            dA = self.layers[i].backward(dZ)
            grad_W_list.append(self.layers[i].grad_W.copy())
            grad_b_list.append(self.layers[i].grad_b.copy())

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    @staticmethod
    def _to_onehot(y, num_classes=10):
        y = y.astype(int)
        out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
        out[np.arange(y.shape[0]), y] = 1.0
        return out

    def predict_logits(self, X, batch_size=1024):
        logits_all = []
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            logits_all.append(self.forward(X[start:end]))
        return np.vstack(logits_all)

    def predict(self, X, batch_size=1024):
        logits = self.predict_logits(X, batch_size=batch_size)
        return np.argmax(logits, axis=1)

    def evaluate(self, X, y, batch_size=1024):
        logits = self.predict_logits(X, batch_size=batch_size)
        y_onehot = self._to_onehot(y, self.output_dim)
        if self.loss_name == "cross_entropy":
            loss, _ = cross_entropy_from_logits(logits, y_onehot)
        else:
            loss, _ = mse_from_logits(logits, y_onehot)

        y_pred = np.argmax(logits, axis=1)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="macro", zero_division=0
        )

        return {
            "logits": logits,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def train(
        self,
        X_train,
        y_train,
        epochs=1,
        batch_size=32,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        wandb_run=None,
    ):
        y_train_onehot = self._to_onehot(y_train, self.output_dim)
        n_samples = X_train.shape[0]
        history = []

        for epoch in range(1, epochs + 1):
            indices = self.rng.permutation(n_samples)
            epoch_losses = []
            grad_norms_first_layer = []
            zero_activation_fractions = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train_onehot[batch_idx]

                logits = self.forward(X_batch)
                self.backward(y_batch, logits)
                self.update_weights()

                if self.last_loss is not None:
                    epoch_losses.append(self.last_loss)
                grad_norms_first_layer.append(float(np.linalg.norm(self.layers[0].grad_W)))
                if self.hidden_activations:
                    zero_fraction = float(np.mean(self.hidden_activations[0] == 0.0))
                    zero_activation_fractions.append(zero_fraction)

            train_metrics_full = self.evaluate(X_train, y_train, batch_size=batch_size)
            train_metrics = {k: v for k, v in train_metrics_full.items() if k != "logits"}
            val_metrics = None
            test_metrics = None
            if X_val is not None and y_val is not None:
                val_metrics_full = self.evaluate(X_val, y_val, batch_size=batch_size)
                val_metrics = {k: v for k, v in val_metrics_full.items() if k != "logits"}
            if X_test is not None and y_test is not None:
                test_metrics_full = self.evaluate(X_test, y_test, batch_size=batch_size)
                test_metrics = {k: v for k, v in test_metrics_full.items() if k != "logits"}

            epoch_record = {
                "epoch": epoch,
                "train_loss_step_mean": float(np.mean(epoch_losses)) if epoch_losses else np.nan,
                "grad_norm_first_layer_mean": (
                    float(np.mean(grad_norms_first_layer)) if grad_norms_first_layer else np.nan
                ),
                "dead_neuron_fraction_layer1_mean": (
                    float(np.mean(zero_activation_fractions)) if zero_activation_fractions else 0.0
                ),
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            }
            history.append(epoch_record)

            if wandb_run is not None:
                payload = {
                    "epoch": epoch,
                    "learning_rate": self.learning_rate,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/precision": train_metrics["precision"],
                    "train/recall": train_metrics["recall"],
                    "train/f1": train_metrics["f1"],
                    "train/grad_norm_first_layer": epoch_record["grad_norm_first_layer_mean"],
                    "train/dead_neuron_fraction_layer1": epoch_record[
                        "dead_neuron_fraction_layer1_mean"
                    ],
                }

                if val_metrics is not None:
                    payload.update(
                        {
                            "val/loss": val_metrics["loss"],
                            "val/accuracy": val_metrics["accuracy"],
                            "val/precision": val_metrics["precision"],
                            "val/recall": val_metrics["recall"],
                            "val/f1": val_metrics["f1"],
                        }
                    )
                if test_metrics is not None:
                    payload.update(
                        {
                            "test/loss": test_metrics["loss"],
                            "test/accuracy": test_metrics["accuracy"],
                            "test/precision": test_metrics["precision"],
                            "test/recall": test_metrics["recall"],
                            "test/f1": test_metrics["f1"],
                        }
                    )
                wandb_run.log(payload)

        return history

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
