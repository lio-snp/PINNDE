from __future__ import annotations

import numpy as np
import torch

from pinnde.model import ConditionalPINN
from pinnde.problem import DEFAULT_PROBLEM, analytical_solution, normalize_xi, normalize_z
from pinnde.training import compute_residual


def test_analytical_solution_matches_initial_position() -> None:
    xi_values = np.array([0.1, 0.2, 0.3, 0.4])
    values = analytical_solution(np.zeros_like(xi_values), xi_values)
    assert np.allclose(values, DEFAULT_PROBLEM.x0)


def test_analytical_solution_matches_initial_velocity() -> None:
    xi_values = np.array([0.1, 0.2, 0.3, 0.4])
    step = 1e-6
    forward = analytical_solution(step, xi_values)
    backward = analytical_solution(-step, xi_values)
    numerical_velocity = (forward - backward) / (2.0 * step)
    assert np.allclose(numerical_velocity, DEFAULT_PROBLEM.v0, atol=1e-5)


def test_normalization_maps_bounds_correctly() -> None:
    z = torch.tensor([[DEFAULT_PROBLEM.z_min], [DEFAULT_PROBLEM.z_max]])
    xi = torch.tensor([[DEFAULT_PROBLEM.xi_min], [DEFAULT_PROBLEM.xi_max]])
    assert torch.allclose(normalize_z(z), torch.tensor([[-1.0], [1.0]]))
    assert torch.allclose(normalize_xi(xi), torch.tensor([[-1.0], [1.0]]))


def test_residual_shapes_are_consistent() -> None:
    model = ConditionalPINN()
    z = torch.rand(16, 1, requires_grad=True)
    xi = torch.full((16, 1), 0.2)
    x, dx_dz, d2x_dz2, residual = compute_residual(model, z, xi)
    assert x.shape == (16, 1)
    assert dx_dz.shape == (16, 1)
    assert d2x_dz2.shape == (16, 1)
    assert residual.shape == (16, 1)
