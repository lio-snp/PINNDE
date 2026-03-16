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
    DEFAULT_CURRICULUM,
    DEFAULT_PROBLEM,
    ConditionalPINN,
    ModelConfig,
    check_initial_conditions,
    checkpoint_payload,
    compute_residual_summary,
    curriculum_as_dict,
    evaluate_error,
    fine_tune_adaptive,
    fine_tune_hardcase,
    fine_tune_tail_weighted,
    history_as_dict,
    resolve_device,
    set_seed,
    summarize_metrics,
    train_curriculum,
)
from pinnde.plots import save_error_curves, save_prediction_plot, save_solution_heatmap, save_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the best PINN pipeline from scratch.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_PROBLEM.seed)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "full_run")
    parser.add_argument("--checkpoint-name", default="pinnde_best.pt")
    parser.add_argument("--skip-adaptive", action="store_true")
    parser.add_argument("--skip-hardcase", action="store_true")
    parser.add_argument("--skip-tail-weighted", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    print(f"Using device: {device}")

    output_dir: Path = args.output_dir
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model = ConditionalPINN(config=ModelConfig(), spec=DEFAULT_PROBLEM).to(device)

    curriculum_history = train_curriculum(model, device=device, spec=DEFAULT_PROBLEM)
    save_training_history(
        curriculum_history,
        figures_dir / "training_curriculum.png",
        title="Curriculum training loss",
    )

    adaptive_history = {"loss": [], "phys": [], "ic": []}
    if not args.skip_adaptive:
        adaptive_history = fine_tune_adaptive(model, device=device, spec=DEFAULT_PROBLEM)
        save_training_history(
            adaptive_history,
            figures_dir / "training_adaptive.png",
            title="Adaptive fine-tuning loss",
        )

    hardcase_history = {"loss": [], "phys": [], "ic": []}
    if not args.skip_hardcase:
        hardcase_history = fine_tune_hardcase(model, device=device, spec=DEFAULT_PROBLEM)
        save_training_history(
            hardcase_history,
            figures_dir / "training_hardcase.png",
            title="Hard-case fine-tuning loss",
        )

    tail_weighted_history = {"loss": [], "phys": [], "ic": []}
    if not args.skip_tail_weighted:
        tail_weighted_history = fine_tune_tail_weighted(model, device=device, spec=DEFAULT_PROBLEM)
        save_training_history(
            tail_weighted_history,
            figures_dir / "training_tail_weighted.png",
            title="Tail-weighted fine-tuning loss",
        )

    metrics = {
        "per_xi": evaluate_error(model, device=device),
        "summary": {},
        "initial_conditions": check_initial_conditions(model, device=device),
        "residual_summary": compute_residual_summary(model, device=device),
    }
    metrics["summary"] = summarize_metrics(metrics["per_xi"])

    save_prediction_plot(model, device=device, path=figures_dir / "predictions.png")
    save_error_curves(model, device=device, path=figures_dir / "absolute_error.png")
    save_solution_heatmap(model, device=device, path=figures_dir / "solution_heatmap.png")

    checkpoint_path = output_dir / args.checkpoint_name
    torch.save(
        checkpoint_payload(
            model,
            stage="full_pipeline",
            extra={
                "seed": args.seed,
                "curriculum": curriculum_as_dict(DEFAULT_CURRICULUM),
                "skip_adaptive": args.skip_adaptive,
                "skip_hardcase": args.skip_hardcase,
                "skip_tail_weighted": args.skip_tail_weighted,
            },
        ),
        checkpoint_path,
    )

    run_payload = {
        "device": str(device),
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "histories": {
            "curriculum": history_as_dict(curriculum_history),
            "adaptive": history_as_dict(adaptive_history),
            "hardcase": history_as_dict(hardcase_history),
            "tail_weighted": history_as_dict(tail_weighted_history),
        },
    }
    (metrics_dir / "training_run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics["summary"], indent=2))
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
