from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ProblemSpec:
    x0: float = 0.7
    v0: float = 1.2
    xi_min: float = 0.1
    xi_max: float = 0.4
    z_min: float = 0.0
    z_max: float = 20.0
    seed: int = 42


DEFAULT_PROBLEM = ProblemSpec()
DEFAULT_EVAL_XI_VALUES = (0.1, 0.2, 0.3, 0.4)


def set_seed(seed: int = DEFAULT_PROBLEM.seed) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def analytical_solution(
    z: np.ndarray | list[float] | float,
    xi: np.ndarray | list[float] | float,
    spec: ProblemSpec = DEFAULT_PROBLEM,
) -> np.ndarray:
    z_np = np.asarray(z, dtype=np.float64)
    xi_np = np.asarray(xi, dtype=np.float64)
    wd = np.sqrt(np.maximum(1.0 - xi_np**2, 1e-12))
    c1 = spec.x0
    c2 = (spec.v0 + xi_np * spec.x0) / wd
    return np.exp(-xi_np * z_np) * (
        c1 * np.cos(wd * z_np) + c2 * np.sin(wd * z_np)
    )


def normalize_z(z: torch.Tensor, spec: ProblemSpec = DEFAULT_PROBLEM) -> torch.Tensor:
    return 2.0 * z / spec.z_max - 1.0


def normalize_xi(xi: torch.Tensor, spec: ProblemSpec = DEFAULT_PROBLEM) -> torch.Tensor:
    return 2.0 * (xi - spec.xi_min) / (spec.xi_max - spec.xi_min) - 1.0


def problem_metadata(spec: ProblemSpec = DEFAULT_PROBLEM) -> dict[str, float]:
    return asdict(spec)
