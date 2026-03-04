# Assignment 1: NumPy MLP for MNIST/Fashion-MNIST

NumPy-only implementation of a configurable Multi-Layer Perceptron for DA6401 Assignment 1.

## Submission Links

- GitHub Repository: [MiRL-IITM/da6401_assignment_1](https://github.com/MiRL-IITM/da6401_assignment_1) (replace with your fork URL if required by submission)
- W&B Project: [da6401_assignment_1](https://wandb.ai/da25s007-/da6401_assignment_1)
- W&B Report: `ADD_PUBLIC_WANDB_REPORT_LINK_HERE` (replace before final submission)

## Implemented Requirements (Part 1)

- Configurable MLP with logits output (no output softmax in `forward`)
- Hidden activations: `relu`, `sigmoid`, `tanh`
- Losses from logits: `cross_entropy`, `mean_squared_error`
- Optimizers: `sgd`, `momentum`, `nag`, `rmsprop`
- Weight initialization: `random`, `xavier`, `zeros` (zeros used for symmetry experiment)
- Shared CLI for `src/train.py` and `src/inference.py`
- Layer-level gradients exposed as `layer.grad_W` and `layer.grad_b`
- `NeuralNetwork.backward()` returns gradients from last layer to first
- NumPy serialization with `get_weights()` and `set_weights()`

## Setup

```bash
pip install -r requirements.txt
```

## Default Best Configuration

Current defaults in CLI and artifacts match the selected MNIST best configuration:

- `dataset=mnist`
- `epochs=25`
- `batch_size=64`
- `loss=cross_entropy`
- `optimizer=rmsprop`
- `learning_rate=0.0005`
- `weight_decay=0.0`
- `num_layers=2`
- `hidden_size=128 128`
- `activation=tanh`
- `weight_init=random`

Submission artifacts:

- `src/best_model.npy`
- `src/best_config.json`

## CLI Arguments (Train and Inference)

- `-d`, `-dataset`, `--dataset`: `mnist` | `fashion_mnist`
- `-e`, `-epochs`, `--epochs`
- `-b`, `-batch_size`, `--batch_size`
- `-l`, `-loss`, `--loss`: `cross_entropy` | `mean_squared_error` (alias `mse` also accepted)
- `-o`, `-optimizer`, `--optimizer`: `sgd` | `momentum` | `nag` | `rmsprop`
- `-lr`, `-learning_rate`, `--learning_rate`
- `-wd`, `--weight_decay`
- `-nhl`, `--num_layers`
- `-sz`, `--hidden_size` (space-separated list)
- `-a`, `--activation`: `sigmoid` | `tanh` | `relu`
- `-w_i`, `--weight_init`: `random` | `xavier` | `zeros`
- `-w_p`, `--wandb_project`
- `--wandb_entity`
- `--wandb_mode`: `online` | `offline` | `disabled`
- `--seed`
- `--model_path`
- `--config_path`

## Train

```bash
python src/train.py \
  -d mnist \
  -e 25 \
  -b 64 \
  -l cross_entropy \
  -o rmsprop \
  -lr 0.0005 \
  -wd 0.0 \
  -nhl 2 \
  -sz 128 128 \
  -a tanh \
  -w_i random \
  -w_p da6401_assignment_1
```

To run without W&B logging:

```bash
python src/train.py --wandb_mode disabled
```

## Inference

```bash
python src/inference.py \
  -d mnist \
  -l cross_entropy \
  -o rmsprop \
  -lr 0.0005 \
  -wd 0.0 \
  -nhl 2 \
  -sz 128 128 \
  -a tanh \
  -w_i random \
  --model_path src/best_model.npy
```

Outputs: Accuracy, Precision, Recall, F1-score, and Loss.

## Numerical Gradient Check

```bash
python src/gradient_check.py
```

This script performs finite-difference checks and reports max relative error.

## W&B Sweep (100 Runs)

Notebook workflow:

- Open `notebooks/report_2_2_hyperparameter_sweep.ipynb`
- Set `RUN_SWEEP = True`
- Run all cells to create and launch the sweep agent

Default sweep search space file:

- `src/wandb_sweep_config.yaml`

## Report/Experiment Notebooks

- `notebooks/report_2_1_data_exploration.ipynb`
- `notebooks/report_2_2_hyperparameter_sweep.ipynb`
- `notebooks/report_2_3_optimizer_showdown.ipynb`
- `notebooks/report_2_4_vanishing_gradient.ipynb`
- `notebooks/report_2_5_dead_neuron.ipynb`
- `notebooks/report_2_6_loss_comparison.ipynb`
- `notebooks/report_2_7_global_performance.ipynb`
- `notebooks/report_2_8_error_analysis.ipynb`
- `notebooks/report_2_9_weight_init_symmetry.ipynb`
- `notebooks/report_2_10_fashion_transfer.ipynb`

## Notes

- `sgd` is implemented as mini-batch gradient descent updates over batches.
- `best_config.json` stores hyperparameters, seed, timestamp, and final validation/test metrics.
- Replace `ADD_PUBLIC_WANDB_REPORT_LINK_HERE` with your public W&B report URL before final submission.
