# SetPINNs

SetPINNs explores physics-informed neural networks (PINNs) through a set-based transformer architecture. Instead of treating each collocation point independently, SetPINNs groups points into small unordered sets and applies self-attention to exchange information within the set. This repository accompanies experiments on several partial differential equations (PDEs), contrasting SetPINNs with baseline PINN variants and providing reusable training utilities.

## Why SetPINNs
- Works on unordered point sets and preserves permutation invariance via a lightweight transformer encoder (`src/setpinn/models/setpinns.py`).
- Supports Element-Aware Sampling (EAS) to draw structured mini-sets of collocation points (`src/setpinn/data.py`).
- Retains the standard PINN training recipe (boundary + residual losses, automatic differentiation) while adding strong attention-based priors.

## Repository Layout
- `experiments/train.py` — main training script for PINN variants, including SetPINNs.
- `experiments/*.py` — additional experiment scripts (landscape visualisations, domain-specific demos).
- `src/setpinn/models/` — model definitions: SetPINNs, classic PINNs, PiNNsFormers, FLS, QRes, KAN, etc.
- `src/setpinn/data.py` — data generation and sampling utilities for PDE collocation, boundary, and test grids.
- `src/setpinn/losses.py` — physics-informed loss terms for wave, reaction, convection, plate, etc.
- `src/setpinn/util.py` — experiment helpers (seeding, logging, plotting, experiment directory creation).

## Installation
1. Use Python 3.10 (the project is configured for Poetry; pip also works).
2. Create an environment and install dependencies:
   ```bash
   # With Poetry
   poetry install

   # Or with pip
   pip install -e .  # installs torch, numpy, matplotlib, pandas, tqdm, scikit-learn
   ```

## Quick Start
Train SetPINNs on the harmonic equation (saves outputs under `./runs`):
```bash
poetry run python experiments1/train.py \
  --exp_path ./runs \
  --exp_name setpinns-harmonic-norm \
  --device cuda:0 \
  --res_points 50 \
  --test_points 101 \
  --training_iterations 500 \
  --use_adam_warmup \
  --adam_steps 20000 \
  --warmup_steps 2000
```

Key flags:
- `--exp_path` controls where experiment artifacts are stored.
- `--device` accepts `cuda:<idx>` or `cpu`.
- `--res_points`/`--test_points` set the resolution of the interior and evaluation grids.
- `--use_adam_warmup` toggles an Adam warmup phase before LBFGS fine-tuning; adjust `--adam_steps`, `--warmup_steps`, `--adam_lr`, and betas as needed.
- `--training_iterations` specifies LBFGS outer iterations (each iteration runs a multi-step line search).

During training the script prints progress bars for Adam and LBFGS. The best-performing weights (lowest total loss) are checkpointed as `best_model.pth`.

## Data Pipeline
- `get_dataset` creates collocation (`x_res`, `t_res`), boundary (`x_left`, `x_right`, `x_upper`, `x_lower`), and test grids directly from analytical ranges defined in `X_Y_RANGES`.
- For SetPINNs, EAS partitions the spatial-temporal mesh into sub-squares and samples `set_size` points per cell (default `set_size=4` in `experiments/train.py`). The resulting tensors have shape `[B, S, 1]`, matching the transformer-friendly set representation.
- Boundary and test tensors are automatically converted to PyTorch tensors with gradients enabled for autograd-based residual calculation.
- Analytical solutions in `src/setpinn/analytical_sol.py` provide closed-form references for error metrics and plotting.

## Model Architecture
- Input points `(x, t)` are first embedded with a linear projection.
- A stack of pre-norm transformer encoder layers equipped with custom multi-head self-attention (`MultiHeadSelfAttention`) shares information within each set; this keeps higher-order derivatives differentiable.
- The prediction head maps the encoded features to `u(x, t)`; gradients and Hessians with respect to inputs are taken via PyTorch autograd for residual losses.

Baseline models in this repo (classic PINNs, PINNsformer, QRes, KAN, etc.) use the same training and evaluation utilities, making it easy to compare architectures by adjusting `exp_name`.

## Training Outputs
Each run under `exp_path/<equation>/<model>/<technique>/seed_<seed>/` contains:
- `best_model.pth` — weights with the lowest observed loss.
- `logging.pkl` — pickle with training losses, runtime, predictions, and metadata (parameter count).
- `errors.txt` — relative L1/L2 error summary on the test grid.
- `<exp_name>_pred.pdf`, `<exp_name>_exact.pdf`, `<exp_name>_error.pdf` — heatmaps of predicted field, analytical solution, and absolute error.

## License
This paper is under review. Do not use this code for personal use without author's permission.
