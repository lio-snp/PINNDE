from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from pinnde import (
    check_initial_conditions,
    checkpoint_payload,
    compute_residual_summary,
    evaluate_error,
    fine_tune_hardcase,
    fine_tune_tail_weighted,
    history_as_dict,
    load_checkpoint,
    resolve_device,
    set_seed,
    summarize_metrics,
)
from pinnde.plots import save_error_curves, save_prediction_plot, save_solution_heatmap, save_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine an existing checkpoint with targeted fine-tuning.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "pinnde_v6_model.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["hardcase", "tail-weighted"], default="tail-weighted")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "refined")
    parser.add_argument("--checkpoint-name", default="pinnde_refined.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint} on {device}")

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "hardcase":
        history = fine_tune_hardcase(
            model,
            device=device,
            epochs=args.epochs or 1200,
        )
    else:
        history = fine_tune_tail_weighted(
            model,
            device=device,
            epochs=args.epochs or 600,
        )

    save_training_history(history, figures_dir / f"{args.mode}_history.png", title=f"{args.mode} fine-tuning")

    per_xi = evaluate_error(model, device=device)
    payload = {
        "source_checkpoint": str(args.checkpoint),
        "source_metadata": metadata,
        "mode": args.mode,
        "per_xi": per_xi,
        "summary": summarize_metrics(per_xi),
        "initial_conditions": check_initial_conditions(model, device=device),
        "residual_summary": compute_residual_summary(model, device=device),
        "history": history_as_dict(history),
    }

    save_prediction_plot(model, device=device, path=figures_dir / "predictions.png")
    save_error_curves(model, device=device, path=figures_dir / "absolute_error.png")
    save_solution_heatmap(model, device=device, path=figures_dir / "solution_heatmap.png")

    checkpoint_path = output_dir / args.checkpoint_name
    torch.save(
        checkpoint_payload(
            model,
            stage=args.mode,
            extra={"source_checkpoint": str(args.checkpoint)},
        ),
        checkpoint_path,
    )
    metrics_path = metrics_dir / f"{checkpoint_path.stem}_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved refined checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
