from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluation import predict
from .model import ConditionalPINN
from .problem import DEFAULT_EVAL_XI_VALUES, DEFAULT_PROBLEM, ProblemSpec, analytical_solution


def save_training_history(history: dict[str, list[float]], path: Path, title: str) -> None:
    if not history.get("loss"):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.semilogy(history["loss"], label="Total loss")
    if history.get("phys"):
        plt.semilogy(history["phys"], label="Physics loss")
    if history.get("ic"):
        plt.semilogy(history["ic"], label="IC loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_prediction_plot(
    model: ConditionalPINN,
    device: torch.device,
    path: Path,
    xi_values: tuple[float, ...] = DEFAULT_EVAL_XI_VALUES,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    num_points: int = 500,
) -> None:
    z = np.linspace(spec.z_min, spec.z_max, num_points)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for xi in xi_values:
        y_true = analytical_solution(z, xi, spec=spec)
        y_pred = predict(model, z, xi, device=device)
        plt.plot(z, y_true, "--", label=f"Analytical, xi={xi:.2f}")
        plt.plot(z, y_pred, label=f"PINN, xi={xi:.2f}")
    plt.xlabel("z")
    plt.ylabel("x(z)")
    plt.title("PINN vs analytical solution")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_error_curves(
    model: ConditionalPINN,
    device: torch.device,
    path: Path,
    xi_values: tuple[float, ...] = DEFAULT_EVAL_XI_VALUES,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    num_points: int = 500,
) -> None:
    z = np.linspace(spec.z_min, spec.z_max, num_points)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for xi in xi_values:
        y_true = analytical_solution(z, xi, spec=spec)
        y_pred = predict(model, z, xi, device=device)
        plt.plot(z, np.abs(y_pred - y_true), label=f"xi={xi:.2f}")
    plt.xlabel("z")
    plt.ylabel("Absolute error")
    plt.title("Absolute error curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_solution_heatmap(
    model: ConditionalPINN,
    device: torch.device,
    path: Path,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    num_z: int = 300,
    num_xi: int = 100,
) -> None:
    z_grid = np.linspace(spec.z_min, spec.z_max, num_z)
    xi_grid = np.linspace(spec.xi_min, spec.xi_max, num_xi)
    z_mesh, xi_mesh = np.meshgrid(z_grid, xi_grid)

    z_tensor = torch.tensor(z_mesh.reshape(-1, 1), dtype=torch.float32, device=device)
    xi_tensor = torch.tensor(xi_mesh.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(z_tensor, xi_tensor).cpu().numpy().reshape(num_xi, num_z)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.imshow(
        pred,
        aspect="auto",
        origin="lower",
        extent=[spec.z_min, spec.z_max, spec.xi_min, spec.xi_max],
    )
    plt.colorbar(label="x(z, xi)")
    plt.xlabel("z")
    plt.ylabel("xi")
    plt.title("PINN solution heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
