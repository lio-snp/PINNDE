from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .problem import DEFAULT_PROBLEM, ProblemSpec, normalize_xi, normalize_z


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    num_hidden: int = 4
    num_bands: int = 4


class FourierFeatures(nn.Module):
    def __init__(self, num_bands: int = 4) -> None:
        super().__init__()
        self.num_bands = num_bands
        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def forward(self, z_norm: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.view(1, -1)
        z_proj = torch.pi * z_norm * freqs
        return torch.cat(
            [z_norm, xi_norm, torch.sin(z_proj), torch.cos(z_proj), z_norm * xi_norm],
            dim=1,
        )


class ConditionalPINN(nn.Module):
    def __init__(
        self,
        config: ModelConfig = ModelConfig(),
        spec: ProblemSpec = DEFAULT_PROBLEM,
    ) -> None:
        super().__init__()
        self.config = config
        self.spec = spec
        self.encoder = FourierFeatures(num_bands=config.num_bands)

        in_dim = 2 + 2 * config.num_bands + 1
        layers: list[nn.Module] = [nn.Linear(in_dim, config.hidden_dim), nn.Tanh()]
        for _ in range(config.num_hidden - 1):
            layers.extend([nn.Linear(config.hidden_dim, config.hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        z_norm = normalize_z(z, self.spec)
        xi_norm = normalize_xi(xi, self.spec)
        features = self.encoder(z_norm, xi_norm)
        return self.net(features)


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> ModelConfig:
    weight_keys = [key for key in state_dict if key.startswith("net.") and key.endswith(".weight")]
    sorted_keys = sorted(weight_keys, key=lambda key: int(key.split(".")[1]))
    first_weight = state_dict[sorted_keys[0]]
    hidden_dim = int(first_weight.shape[0])
    in_dim = int(first_weight.shape[1])
    num_hidden = len(sorted_keys) - 1
    num_bands = (in_dim - 3) // 2
    return ModelConfig(hidden_dim=hidden_dim, num_hidden=num_hidden, num_bands=num_bands)


def checkpoint_payload(
    model: ConditionalPINN,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "stage": stage,
    }
    if extra:
        payload["extra"] = extra
    return payload


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    model_config: ModelConfig | None = None,
    spec: ProblemSpec = DEFAULT_PROBLEM,
) -> tuple[ConditionalPINN, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    metadata: dict[str, Any]

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
        if model_config is None:
            config_dict = checkpoint.get("model_config")
            if config_dict:
                model_config = ModelConfig(**config_dict)
    else:
        state_dict = checkpoint
        metadata = {}

    if model_config is None:
        model_config = infer_model_config(state_dict)

    model = ConditionalPINN(config=model_config, spec=spec).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model, metadata
