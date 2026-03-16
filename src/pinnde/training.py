from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .model import ConditionalPINN, ModelConfig
from .problem import DEFAULT_PROBLEM, ProblemSpec


History = dict[str, list[float]]


@dataclass(frozen=True)
class CurriculumStage:
    z_upper: float
    epochs: int
    lambda_ic: float
    n_interior: int
    n_initial: int


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_CURRICULUM = (
    CurriculumStage(z_upper=5.0, epochs=2000, lambda_ic=30.0, n_interior=1024, n_initial=256),
    CurriculumStage(z_upper=10.0, epochs=2000, lambda_ic=20.0, n_interior=1024, n_initial=256),
    CurriculumStage(z_upper=15.0, epochs=2000, lambda_ic=20.0, n_interior=1536, n_initial=256),
    CurriculumStage(z_upper=20.0, epochs=3000, lambda_ic=15.0, n_interior=2048, n_initial=256),
)


def make_empty_history() -> History:
    return {"loss": [], "phys": [], "ic": []}


def history_as_dict(history: History) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.items()}


def curriculum_as_dict(stages: tuple[CurriculumStage, ...]) -> list[dict[str, float | int]]:
    return [asdict(stage) for stage in stages]


def sample_initial(
    n_samples: int,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
) -> tuple[torch.Tensor, torch.Tensor]:
    z0 = torch.zeros(n_samples, 1, device=device)
    xi = torch.rand(n_samples, 1, device=device) * (spec.xi_max - spec.xi_min) + spec.xi_min
    z0.requires_grad_(True)
    return z0, xi


def sample_interior_biased(
    n_samples: int,
    z_upper: float,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.rand(n_samples, 1, device=device)
    z = z_upper * torch.sqrt(u)

    u2 = torch.rand(n_samples, 1, device=device)
    xi = spec.xi_min + (spec.xi_max - spec.xi_min) * (u2 ** 1.5)

    z.requires_grad_(True)
    return z, xi


def sample_interior_hardcase(
    n_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = 8.0 + 12.0 * torch.rand(n_samples, 1, device=device)
    xi = 0.10 + 0.06 * torch.rand(n_samples, 1, device=device)
    z.requires_grad_(True)
    return z, xi


def compute_residual(
    model: ConditionalPINN,
    z: torch.Tensor,
    xi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = model(z, xi)
    dx_dz = torch.autograd.grad(
        x,
        z,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2x_dz2 = torch.autograd.grad(
        dx_dz,
        z,
        grad_outputs=torch.ones_like(dx_dz),
        create_graph=True,
        retain_graph=True,
    )[0]
    residual = d2x_dz2 + 2.0 * xi * dx_dz + x
    return x, dx_dz, d2x_dz2, residual


def build_residual_pool(
    model: ConditionalPINN,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    num_z: int = 200,
    num_xi: int = 80,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_vals = torch.linspace(spec.z_min, spec.z_max, num_z, device=device).view(-1, 1)
    xi_vals = torch.linspace(spec.xi_min, spec.xi_max, num_xi, device=device).view(-1, 1)

    zz, xx = torch.meshgrid(z_vals.squeeze(), xi_vals.squeeze(), indexing="ij")
    z_pool = zz.reshape(-1, 1).clone().detach().requires_grad_(True)
    xi_pool = xx.reshape(-1, 1).clone().detach()

    _, _, _, residual = compute_residual(model, z_pool, xi_pool)
    residual_abs = residual.detach().abs().reshape(-1)
    return z_pool.detach(), xi_pool.detach(), residual_abs


def sample_interior_adaptive(
    model: ConditionalPINN,
    n_samples: int,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    mix_ratio: float = 0.5,
    hard_ratio: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_adaptive = int(n_samples * mix_ratio)
    n_hard = int(n_samples * hard_ratio)
    n_random = n_samples - n_adaptive - n_hard

    z_pool, xi_pool, residual_abs = build_residual_pool(model, device=device, spec=spec)
    weights = residual_abs + 1e-8
    weights = weights / weights.sum()

    indices = torch.multinomial(weights, n_adaptive, replacement=True)
    z_adaptive = z_pool[indices].clone().detach()
    xi_adaptive = xi_pool[indices].clone().detach()

    z_hard, xi_hard = sample_interior_hardcase(n_hard, device=device)
    z_hard = z_hard.detach()
    xi_hard = xi_hard.detach()

    z_random, xi_random = sample_interior_biased(
        n_random,
        z_upper=spec.z_max,
        device=device,
        spec=spec,
    )
    z_random = z_random.detach()
    xi_random = xi_random.detach()

    z = torch.cat([z_adaptive, z_hard, z_random], dim=0).requires_grad_(True)
    xi = torch.cat([xi_adaptive, xi_hard, xi_random], dim=0)
    return z, xi


def ic_loss(
    model: ConditionalPINN,
    n_initial: int,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
) -> tuple[torch.Tensor, torch.Tensor]:
    z0, xi0 = sample_initial(n_initial, device=device, spec=spec)
    x0_pred = model(z0, xi0)
    dx0_dz = torch.autograd.grad(
        x0_pred,
        z0,
        grad_outputs=torch.ones_like(x0_pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    loss_x0 = torch.mean((x0_pred - spec.x0) ** 2)
    loss_v0 = torch.mean((dx0_dz - spec.v0) ** 2)
    return loss_x0 + loss_v0, dx0_dz


def stage_loss(
    model: ConditionalPINN,
    n_interior: int,
    n_initial: int,
    z_upper: float,
    lambda_ic: float,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_r, xi_r = sample_interior_biased(n_interior, z_upper=z_upper, device=device, spec=spec)
    _, _, _, residual = compute_residual(model, z_r, xi_r)
    loss_phys = torch.mean(residual ** 2)
    loss_ic, _ = ic_loss(model, n_initial=n_initial, device=device, spec=spec)
    loss = loss_phys + lambda_ic * loss_ic
    return loss, loss_phys.detach(), loss_ic.detach()


def adaptive_loss(
    model: ConditionalPINN,
    n_interior: int,
    n_initial: int,
    lambda_ic: float,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_r, xi_r = sample_interior_adaptive(model, n_interior, device=device, spec=spec)
    _, _, _, residual = compute_residual(model, z_r, xi_r)
    loss_phys = torch.mean(residual ** 2)
    loss_ic, _ = ic_loss(model, n_initial=n_initial, device=device, spec=spec)
    loss = loss_phys + lambda_ic * loss_ic
    return loss, loss_phys.detach(), loss_ic.detach()


def hardcase_loss(
    model: ConditionalPINN,
    n_interior: int,
    n_initial: int,
    lambda_ic: float,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_r, xi_r = sample_interior_adaptive(
        model,
        n_interior,
        device=device,
        spec=spec,
        mix_ratio=0.5,
        hard_ratio=0.3,
    )
    _, _, _, residual = compute_residual(model, z_r, xi_r)
    loss_phys = torch.mean(residual ** 2)
    loss_ic, _ = ic_loss(model, n_initial=n_initial, device=device, spec=spec)
    loss = loss_phys + lambda_ic * loss_ic
    return loss, loss_phys.detach(), loss_ic.detach()


def region_weighted_loss(
    model: ConditionalPINN,
    n_interior: int,
    n_initial: int,
    lambda_ic: float,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    tail_emphasis: float = 3.0,
    low_xi_emphasis: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_r, xi_r = sample_interior_adaptive(
        model,
        n_interior,
        device=device,
        spec=spec,
        mix_ratio=0.55,
        hard_ratio=0.25,
    )
    _, _, _, residual = compute_residual(model, z_r, xi_r)

    z_weight = 1.0 + tail_emphasis * (z_r / spec.z_max) ** 2
    xi_weight = 1.0 + low_xi_emphasis * (
        (spec.xi_max - xi_r) / (spec.xi_max - spec.xi_min)
    ) ** 2
    weights = z_weight * xi_weight

    loss_phys = torch.mean(weights * (residual ** 2))
    loss_ic, _ = ic_loss(model, n_initial=n_initial, device=device, spec=spec)
    loss = loss_phys + lambda_ic * loss_ic
    return loss, loss_phys.detach(), loss_ic.detach()


def _run_training_loop(
    model: ConditionalPINN,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    loss_builder: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    history: History,
    log_prefix: str,
    log_interval: int,
) -> History:
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, loss_phys, loss_ic = loss_builder()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"{log_prefix} numerical issue at epoch {epoch}, stopping early.")
            return history

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["phys"].append(float(loss_phys.item()))
        history["ic"].append(float(loss_ic.item()))

        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"{log_prefix} epoch {epoch:5d} | "
                f"total={loss.item():.6e} | "
                f"phys={loss_phys.item():.6e} | "
                f"ic={loss_ic.item():.6e}"
            )
    return history


def train_curriculum(
    model: ConditionalPINN,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    stages: tuple[CurriculumStage, ...] = DEFAULT_CURRICULUM,
    lr: float = 1e-3,
) -> History:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    history = make_empty_history()
    global_epoch = 0

    model.train()
    for stage_id, stage in enumerate(stages, start=1):
        print(
            f"\n=== Curriculum stage {stage_id}: z in [0, {stage.z_upper}] "
            f"for {stage.epochs} epochs ==="
        )
        for stage_epoch in range(1, stage.epochs + 1):
            global_epoch += 1
            optimizer.zero_grad()
            loss, loss_phys, loss_ic = stage_loss(
                model,
                n_interior=stage.n_interior,
                n_initial=stage.n_initial,
                z_upper=stage.z_upper,
                lambda_ic=stage.lambda_ic,
                device=device,
                spec=spec,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Curriculum] numerical issue at global epoch {global_epoch}, stopping.")
                return history

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            history["loss"].append(float(loss.item()))
            history["phys"].append(float(loss_phys.item()))
            history["ic"].append(float(loss_ic.item()))

            if global_epoch % 500 == 0 or stage_epoch == 1:
                print(
                    f"[Curriculum] epoch {global_epoch:5d} | "
                    f"total={loss.item():.6e} | "
                    f"phys={loss_phys.item():.6e} | "
                    f"ic={loss_ic.item():.6e}"
                )
    return history


def fine_tune_adaptive(
    model: ConditionalPINN,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    epochs: int = 1500,
    lr: float = 3e-4,
    n_interior: int = 2048,
    n_initial: int = 256,
    lambda_ic: float = 15.0,
) -> History:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = make_empty_history()
    print("\n=== Adaptive fine-tuning ===")
    return _run_training_loop(
        model,
        optimizer,
        epochs,
        lambda: adaptive_loss(
            model,
            n_interior=n_interior,
            n_initial=n_initial,
            lambda_ic=lambda_ic,
            device=device,
            spec=spec,
        ),
        history,
        log_prefix="[Adaptive]",
        log_interval=250,
    )


def fine_tune_hardcase(
    model: ConditionalPINN,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    epochs: int = 1200,
    lr: float = 1e-4,
    n_interior: int = 2048,
    n_initial: int = 256,
    lambda_ic: float = 15.0,
) -> History:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = make_empty_history()
    print("\n=== Hard-case fine-tuning ===")
    return _run_training_loop(
        model,
        optimizer,
        epochs,
        lambda: hardcase_loss(
            model,
            n_interior=n_interior,
            n_initial=n_initial,
            lambda_ic=lambda_ic,
            device=device,
            spec=spec,
        ),
        history,
        log_prefix="[HardCase]",
        log_interval=200,
    )


def fine_tune_tail_weighted(
    model: ConditionalPINN,
    device: torch.device,
    spec: ProblemSpec = DEFAULT_PROBLEM,
    epochs: int = 600,
    lr: float = 7e-5,
    n_interior: int = 2048,
    n_initial: int = 256,
    lambda_ic: float = 15.0,
) -> History:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = make_empty_history()
    print("\n=== Tail-weighted fine-tuning ===")
    return _run_training_loop(
        model,
        optimizer,
        epochs,
        lambda: region_weighted_loss(
            model,
            n_interior=n_interior,
            n_initial=n_initial,
            lambda_ic=lambda_ic,
            device=device,
            spec=spec,
        ),
        history,
        log_prefix="[TailWeighted]",
        log_interval=100,
    )
