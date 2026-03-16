# Result Summary

## Best checkpoint

- `checkpoints/pinnde_v7_model.pt`
- Derived from `v6` by 400 epochs of tail-weighted refinement.

## Evaluation comparison

| Checkpoint | Mean L2 | Worst L2 | Mean Max Error | Worst Max Error |
| --- | ---: | ---: | ---: | ---: |
| `v6` | 0.01831894 | 0.04726235 | 0.03299680 | 0.08038976 |
| `v7` | 0.01787182 | 0.04315477 | 0.03236330 | 0.07318013 |

## `v7` per-parameter errors

| `xi` | L2 Error | Max Error | Mean Abs Error |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.04315477 | 0.07318013 | 0.03652232 |
| 0.20 | 0.01339126 | 0.02357479 | 0.01135741 |
| 0.30 | 0.00554263 | 0.01303528 | 0.00451364 |
| 0.40 | 0.00939862 | 0.01966299 | 0.00783175 |

## Files

- `results/evaluation_v6/metrics/pinnde_v6_model_metrics.json`
- `results/evaluation_v7/metrics/pinnde_v7_model_metrics.json`
- `results/evaluation_v7/figures/predictions.png`
- `results/evaluation_v7/figures/absolute_error.png`
- `results/evaluation_v7/figures/solution_heatmap.png`
