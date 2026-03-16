from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import ConditionalPINN
from .problem import DEFAULT_EVAL_XI_VALUES, DEFAULT_PROBLEM, ProblemSpec, analytical_solution
from .training import compute_residual


@torch.no_grad()
def predict(
    model: ConditionalPINN,
    z: np.ndarray,
    xi: float,
    device: torch.device,
) -> np.ndarray:
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device).reshape(-1, 1)
    xi_tensor = torch.full_like(z_tensor, fill_value=float(xi))
    return model(z_tensor, xi_tensor).cpu().numpy().reshape(-1)


def evaluate_error(
    model: ConditionalPINN,
    device: torch.device,
    xi_values: tuple[float, ...] = DEFAULT_EVAL_XI_VALUES,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    num_points: int = 500,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    z = np.linspace(spec.z_min, spec.z_max, num_points)

    for xi in xi_values:
        y_pred = predict(model, z, xi, device=device)
        y_true = analytical_solution(z, xi, spec=spec)
        abs_err = np.abs(y_pred - y_true)
        results.append(
            {
                "xi": float(xi),
                "l2_error": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                "max_error": float(np.max(abs_err)),
                "mean_abs_error": float(np.mean(abs_err)),
            }
        )
    return results


def check_initial_conditions(
    model: ConditionalPINN,
    device: torch.device,
    xi_values: tuple[float, ...] = DEFAULT_EVAL_XI_VALUES,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    model.eval()

    for xi in xi_values:
        z0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
        xi_tensor = torch.tensor([[xi]], dtype=torch.float32, device=device)
        x0_pred = model(z0, xi_tensor)
        dx0 = torch.autograd.grad(
            x0_pred,
            z0,
            grad_outputs=torch.ones_like(x0_pred),
            create_graph=False,
            retain_graph=False,
        )[0]
        results.append(
            {
                "xi": float(xi),
                "x0_pred": float(x0_pred.item()),
                "v0_pred": float(dx0.item()),
            }
        )
    return results


def compute_residual_summary(
    model: ConditionalPINN,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    num_z: int = 200,
    num_xi: int = 80,
) -> dict[str, float]:
    z_vals = torch.linspace(spec.z_min, spec.z_max, num_z, device=device).view(-1, 1)
    xi_vals = torch.linspace(spec.xi_min, spec.xi_max, num_xi, device=device).view(-1, 1)
    zz, xx = torch.meshgrid(z_vals.squeeze(), xi_vals.squeeze(), indexing="ij")
    z = zz.reshape(-1, 1).clone().detach().requires_grad_(True)
    xi = xx.reshape(-1, 1).clone().detach()

    _, _, _, residual = compute_residual(model, z, xi)
    residual_abs = residual.detach().abs().cpu().numpy().reshape(-1)
    return {
        "mean_abs_residual": float(np.mean(residual_abs)),
        "max_abs_residual": float(np.max(residual_abs)),
    }


def summarize_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    l2_values = np.array([item["l2_error"] for item in results], dtype=np.float64)
    max_values = np.array([item["max_error"] for item in results], dtype=np.float64)
    return {
        "mean_l2_error": float(np.mean(l2_values)),
        "worst_l2_error": float(np.max(l2_values)),
        "mean_max_error": float(np.mean(max_values)),
        "worst_max_error": float(np.max(max_values)),
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
