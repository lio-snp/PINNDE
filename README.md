# PINNDE GSoC Test

This repository packages my solution to the ML4Sci GSoC test for the GENIE 5 project:
[https://ml4sci.org/gsoc/2026/proposal_GENIE5.html](https://ml4sci.org/gsoc/2026/proposal_GENIE5.html)

The task is to build a Physics-Informed Neural Network in PyTorch for the damped oscillator

`x''(z) + 2 * xi * x'(z) + x(z) = 0`

with:

- `z in [0, 20]`
- `xi in [0.1, 0.4]`
- `x(0) = 0.7`
- `x'(0) = 1.2`

The model learns a conditional solution `x(z, xi)` that generalizes over the damping ratio.

## Approach

The final implementation is the cleaned-up version of the best notebook pipeline:

- Fourier feature encoding for the normalized `z` input.
- Conditional PINN with inputs `(z, xi)`.
- Curriculum training over increasing `z` ranges.
- Residual-adaptive sampling to focus on hard regions.
- Hard-case fine-tuning that emphasizes low-`xi`, long-horizon trajectories.
- Final tail-weighted refinement to further reduce the worst-case error at low `xi`.

## Repository Layout

```text
PINNDE/
├── checkpoints/         # saved model weights
├── notebooks/           # original exploratory notebook
├── results/             # generated figures and metrics
├── scripts/             # train / refine / evaluate entry points
├── src/pinnde/          # reusable package code
└── tests/               # lightweight unit tests
```

## Setup

Use the same environment that has `torch` installed, then from the repo root:

```bash
pip install -e .
```

or:

```bash
pip install -r requirements.txt
```

## Run

Evaluate the current best checkpoint:

```bash
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/pinnde_v7_model.pt
```

Refine an existing checkpoint with targeted fine-tuning:

```bash
python scripts/refine_checkpoint.py --checkpoint checkpoints/pinnde_v6_model.pt --mode tail-weighted
```

Train the full pipeline from scratch:

```bash
python scripts/train_best.py
```

Run the lightweight tests:

```bash
pytest
```

## Notes

- `notebooks/test.ipynb` keeps the original exploration history from `v1` to `v6`.
- `scripts/refine_checkpoint.py` is intended for small targeted improvements on the existing best checkpoint.
- Generated figures and metrics are written under `results/`.

## Current Best Result

The current best checkpoint is `checkpoints/pinnde_v7_model.pt`, obtained by applying a short
tail-weighted fine-tune on top of `v6`.

| Checkpoint | Mean L2 | Worst L2 | Mean Max Error | Worst Max Error |
| --- | ---: | ---: | ---: | ---: |
| `v6` | 0.01832 | 0.04726 | 0.03300 | 0.08039 |
| `v7` | 0.01787 | 0.04315 | 0.03236 | 0.07318 |

Per-`xi` `v7` errors:

| `xi` | L2 Error | Max Error |
| ---: | ---: | ---: |
| 0.10 | 0.04315 | 0.07318 |
| 0.20 | 0.01339 | 0.02357 |
| 0.30 | 0.00554 | 0.01304 |
| 0.40 | 0.00940 | 0.01966 |
