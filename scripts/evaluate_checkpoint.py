from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pinnde import (
    check_initial_conditions,
    compute_residual_summary,
    evaluate_error,
    load_checkpoint,
    resolve_device,
    summarize_metrics,
)
from pinnde.plots import save_error_curves, save_prediction_plot, save_solution_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved PINN checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "pinnde_v6_model.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint} on {device}")

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    per_xi = evaluate_error(model, device=device)
    payload = {
        "checkpoint": str(args.checkpoint),
        "metadata": metadata,
        "per_xi": per_xi,
        "summary": summarize_metrics(per_xi),
        "initial_conditions": check_initial_conditions(model, device=device),
        "residual_summary": compute_residual_summary(model, device=device),
    }

    save_prediction_plot(model, device=device, path=figures_dir / "predictions.png")
    save_error_curves(model, device=device, path=figures_dir / "absolute_error.png")
    save_solution_heatmap(model, device=device, path=figures_dir / "solution_heatmap.png")

    metrics_path = metrics_dir / f"{args.checkpoint.stem}_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
