# DA6401 Assignment 1: NumPy MLP

NumPy-only implementation of a configurable multi-layer perceptron for image classification on MNIST/Fashion-MNIST.

## Links

- github repository: [AkshayAmbekar26/da6401_assignment_1](https://github.com/AkshayAmbekar26/da6401_assignment_1)
- wandb report: [da6401 assignment 1 report](https://wandb.ai/da25s007-/da6401_assignment_1/reports/DA6401-Assignment-01--VmlldzoxNjA3MDIwNg?accessToken=aaj9vs4xrpmqny2kh1hvflkt4fbnmz7vgqmcbxfkqedidc11vmu9juvtooczlh3j)

## What Is Implemented

- configurable MLP using only `numpy`
- output of `forward()` is logits (no output softmax)
- hidden activations: `sigmoid`, `tanh`, `relu`
- losses: `cross_entropy`, `mean_squared_error` (alias: `mse`)
- optimizers: `sgd`, `momentum`, `nag`, `rmsprop`
- initializations: `random`, `xavier`, `zeros`
- shared CLI for `src/train.py` and `src/inference.py`
- per-layer gradients exposed as `layer.grad_W` and `layer.grad_b`
- `NeuralNetwork.backward()` returns gradients from last layer to first

## Repository Layout

- `src/train.py`: training entry point
- `src/inference.py`: model loading + test evaluation
- `src/ann/`: model, layers, activations, losses, optimizers
- `src/utils/cli.py`: shared argument parser
- `src/utils/data_loader.py`: dataset loading + preprocessing
- `src/best_model.npy`: submission model artifact
- `src/best_config.json`: submission config artifact

## Setup

```bash
pip install -r requirements.txt
```

## Shared CLI Arguments

Both `train.py` and `inference.py` accept the same core arguments:

- `-d, --dataset`: `mnist` | `fashion_mnist`
- `-e, --epochs`: number of epochs
- `-b, --batch_size`
- `-l, --loss`: `cross_entropy` | `mean_squared_error` | `mse`
- `-o, --optimizer`: `sgd` | `momentum` | `nag` | `rmsprop`
- `-lr, --learning_rate`
- `-wd, --weight_decay`
- `-nhl, --num_layers`
- `-sz, --hidden_size`: one value or exactly `num_layers` values
- `-a, --activation`: `sigmoid` | `tanh` | `relu`
- `-w_i, --weight_init`: `random` | `xavier` | `zeros`
- `-w_p, --wandb_project`
- `--wandb_entity`
- `--wandb_mode`: `online` | `offline` | `disabled`
- `--seed`
- `--model_path`
- `--config_path`

## Default CLI Configuration

Current defaults in `src/utils/cli.py`:

- `dataset=mnist`
- `epochs=10`
- `batch_size=64`
- `loss=cross_entropy`
- `optimizer=rmsprop`
- `learning_rate=0.0004`
- `weight_decay=0.0001`
- `num_layers=4`
- `hidden_size=128 128 128 128`
- `activation=relu`
- `weight_init=xavier`

## Training

Example:

```bash
python src/train.py \
  --dataset mnist \
  --epochs 10 \
  --batch_size 64 \
  --loss cross_entropy \
  --optimizer rmsprop \
  --learning_rate 0.0004 \
  --weight_decay 0.0001 \
  --num_layers 4 \
  --hidden_size 128 128 128 128 \
  --activation relu \
  --weight_init xavier \
  --wandb_mode disabled
```

Outputs:

- prints final validation/test metrics
- saves model weights (`.npy`) and config (`.json`)
- by default writes to `src/best_model.npy` and `src/best_config.json`

## Inference

```bash
python src/inference.py \
  --dataset mnist \
  --loss cross_entropy \
  --optimizer rmsprop \
  --learning_rate 0.0004 \
  --weight_decay 0.0001 \
  --num_layers 4 \
  --hidden_size 128 128 128 128 \
  --activation relu \
  --weight_init xavier \
  --model_path src/best_model.npy \
  --wandb_mode disabled
```

Printed metrics:

- accuracy
- precision (macro)
- recall (macro)
- f1-score (macro)
- loss

## Submission Artifacts

Final submission files:

- `src/best_model.npy`
- `src/best_config.json`

Current selected model in these artifacts:

- `0.60 * ft_v4_rot25_balanced + 0.40 * ft_v2_inv20_rot35_blur15` (single averaged weight set)

## Gradient Check

```bash
python src/gradient_check.py
```

Expected behavior:

- finite-difference gradient check passes with very low relative error

## Autograder Alignment Notes

- no torch/tensorflow/jax or autograd used in the model math
- `forward()` returns logits only
- per-layer `grad_W` and `grad_b` are available after `backward()`
- backward gradient order is output-layer to input-layer
- both training and inference use argparse-based CLI
