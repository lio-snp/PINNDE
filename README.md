# PINNDE for the ML4Sci GSoC Test

## Overview

This repository contains a PyTorch implementation of a physics-informed neural network for the damped oscillator test in the ML4Sci GSoC 2026 GENIE 5 project.

The target equation is

\[
\frac{d^2 x}{dz^2} + 2 \xi \frac{dx}{dz} + x = 0,
\]

with

- \( z \in [0, 20] \)
- \( \xi \in [0.1, 0.4] \)
- \( x(0) = 0.7 \)
- \( x'(0) = 1.2 \)

The objective is to learn a single conditional model \( x(z, \xi) \) that remains accurate across the full damping interval.

## Problem Formulation

For \( \xi \in [0.1, 0.4] \), the system is underdamped. The analytical solution is available and is used only for evaluation,

\[
x(z, \xi) = e^{-\xi z}
\left[
x_0 \cos(\omega_d z) +
\frac{v_0 + \xi x_0}{\omega_d}\sin(\omega_d z)
\right],
\qquad
\omega_d = \sqrt{1-\xi^2}.
\]

Training is driven by the physics residual

\[
r(z,\xi) = x_{zz}(z,\xi) + 2\xi x_z(z,\xi) + x(z,\xi),
\]

together with the initial conditions. The main challenge is not the existence of the solution, but stable approximation over a long horizon when \( \xi \) is small and oscillations decay slowly.

## Final Model

The final implementation is a conditional PINN with the following components:

- normalized inputs for \( z \) and \( \xi \)
- Fourier features on the normalized \( z \) coordinate
- multilayer perceptron with Tanh activations
- curriculum training over increasing \( z \) ranges
- biased interior sampling toward large \( z \) and small \( \xi \)
- residual-adaptive sampling
- targeted refinement on the low-\( \xi \), long-horizon region

The final code is organized under `src/pinnde/`, with training and evaluation entry points in `scripts/`.

## Iterative Development from v1 to v7

The notebook in `notebooks/test.ipynb` records seven iterations. Each version was motivated by a concrete failure mode observed in the previous stage.

| Version | Main change | Reason for the change | Outcome |
| --- | --- | --- | --- |
| v1 | Baseline conditional MLP with soft initial-condition loss and uniform sampling | Establish a reference implementation and identify the dominant error regime | Stable, but weak long-range accuracy. Error concentrated at low \( \xi \), especially \( \xi = 0.1 \) |
| v2 | Hard initial-condition enforcement through a trial solution, Sobol sampling, SIREN option, Adam then L-BFGS | Remove initial-condition drift and improve oscillatory representation | Optimization became unstable. The run produced NaNs and was discarded |
| v3 | Stabilized hard-constraint model with a reduced corrective term | Recover numerical stability after v2 divergence | Stable but inaccurate. The constrained ansatz limited expressivity over the full interval |
| v4 | Input normalization, Fourier features, biased sampling, curriculum in \( z \) | Address the low-\( \xi \), large-\( z \) error observed in v1 and v3 | Large improvement. This was the first version with acceptable accuracy over the whole domain |
| v5 | Residual-adaptive sampling and safe L-BFGS polishing | Concentrate training points in regions where the PDE residual remained high | Further reduction in average and worst-case error |
| v6 | Hard-case fine-tuning on low-\( \xi \), long-horizon samples | Residual analysis showed that the hardest region was concentrated near \( \xi \in [0.10, 0.16] \), \( z \in [8, 20] \) | Best notebook-stage model |
| v7 | Tail-weighted refinement on top of v6 | Reduce worst-case error without changing the architecture or retraining from scratch | Final best checkpoint |

### Quantitative trajectory

| Version | Mean L2 error | Worst L2 error | Mean max error | Worst max error | Comment |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 | 9.7413e-02 | 2.5117e-01 | 2.1134e-01 | 5.2621e-01 | Baseline, underfits the hardest regime |
| v2 | not reported | not reported | not reported | not reported | Diverged to NaN during optimization |
| v3 | 5.6620e-01 | 7.0285e-01 | 1.5960e+00 | 1.7922e+00 | Stable but substantially underperforms |
| v4 | 3.3734e-02 | 7.8714e-02 | 6.4947e-02 | 1.2572e-01 | Major improvement from representation and sampling changes |
| v5 | 2.4297e-02 | 6.6240e-02 | 4.2274e-02 | 1.0521e-01 | Residual-adaptive sampling improves local accuracy |
| v6 | 1.8319e-02 | 4.7262e-02 | 3.2997e-02 | 8.0390e-02 | Best model produced in the notebook |
| v7 | 1.7872e-02 | 4.3155e-02 | 3.2363e-02 | 7.3180e-02 | Final best checkpoint |

The dominant source of error throughout the study is the low-damping boundary, especially \( \xi = 0.1 \). This is expected, since the solution remains oscillatory for longer and the network must preserve phase information over the entire interval \( z \in [0, 20] \).

## Final Evaluation

The current best checkpoint is `checkpoints/pinnde_v7_model.pt`, obtained by applying 400 epochs of tail-weighted refinement to `v6`.

### Aggregate metrics

| Checkpoint | Mean L2 | Worst L2 | Mean max error | Worst max error |
| --- | ---: | ---: | ---: | ---: |
| v6 | 0.01831894 | 0.04726235 | 0.03299680 | 0.08038976 |
| v7 | 0.01787182 | 0.04315477 | 0.03236330 | 0.07318013 |

### Per-parameter error for v7

| \( \xi \) | L2 error | Max error | Mean absolute error |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.04315477 | 0.07318013 | 0.03652232 |
| 0.20 | 0.01339126 | 0.02357479 | 0.01135741 |
| 0.30 | 0.00554263 | 0.01303528 | 0.00451364 |
| 0.40 | 0.00939862 | 0.01966299 | 0.00783175 |

The improvement from v6 to v7 is modest in the mean, but meaningful in the worst case. This is the relevant direction for this problem, since the largest errors are localized and structurally tied to the slowest-decaying trajectories.

## Repository Structure

```text
PINNDE/
├── checkpoints/         saved model weights from v1 to v7
├── notebooks/           exploratory notebook with the full iteration history
├── results/             evaluation figures and metric summaries
├── scripts/             reproducible train, refine, and evaluate entry points
├── src/pinnde/          reusable implementation
└── tests/               lightweight unit tests
```

## Reproducibility

Install the package in an environment with PyTorch:

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

Evaluate the best checkpoint:

```bash
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/pinnde_v7_model.pt
```

Refine `v6` into `v7`:

```bash
python scripts/refine_checkpoint.py --checkpoint checkpoints/pinnde_v6_model.pt --mode tail-weighted
```

Train the full pipeline from scratch:

```bash
python scripts/train_best.py
```

Run the unit tests:

```bash
pytest
```

## Files of Interest

- `results/evaluation_v6/metrics/pinnde_v6_model_metrics.json`
- `results/evaluation_v7/metrics/pinnde_v7_model_metrics.json`
- `results/evaluation_v7/figures/predictions.png`
- `results/evaluation_v7/figures/absolute_error.png`
- `results/evaluation_v7/figures/solution_heatmap.png`
