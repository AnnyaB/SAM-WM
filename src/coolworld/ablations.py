from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .samwm import SAMWorldModel


class IdentityMentalMap(nn.Module):
    """Ablation: remove sparse state-dependent message passing."""

    def forward(self, z: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:  # noqa: ARG002
        return z


class ZeroExchange(nn.Module):
    """Ablation: remove the conservative pair-exchange operator."""

    def forward(
        self,
        temp: Tensor,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:  # noqa: ARG002
        return torch.zeros_like(temp), temp.new_zeros(())


SAM_ABLATIONS = (
    "samwm",
    "samwm_no_sigreg",
    "samwm_no_exchange",
    "samwm_no_mental_map",
    "samwm_no_residual",
    "samwm_no_rh",
)


def build_samwm(resolved_cfg: dict, variant: str) -> SAMWorldModel:
    if variant not in SAM_ABLATIONS:
        raise ValueError(f"unknown SAM-WM variant {variant!r}")
    model = SAMWorldModel(
        dynamic_dim=int(resolved_cfg.get("dynamic_dim", 3)),
        static_dim=int(resolved_cfg.get("static_dim", 3)),
        hidden_dim=int(resolved_cfg["hidden_dim"]),
        max_source_step_normalized=float(resolved_cfg["max_source_step_normalized"]),
        residual_fraction=float(resolved_cfg["residual_fraction"]),
    )
    if variant == "samwm_no_exchange":
        model.exchange = ZeroExchange()
    elif variant == "samwm_no_mental_map":
        model.mental_map = IdentityMentalMap()
    elif variant == "samwm_no_residual":
        model.residual_fraction = 0.0
    return model


def batch_transform_for_variant(variant: str) -> Callable[[dict[str, Tensor]], dict[str, Tensor]]:
    if variant != "samwm_no_rh":
        return lambda batch: batch

    def no_rh(batch: dict[str, Tensor]) -> dict[str, Tensor]:
        transformed = dict(batch)
        for key in ("context_dynamic", "future_dynamic"):
            value = transformed.get(key)
            if value is None:
                continue
            value = value.clone()
            value[..., 1:] = 0.0
            transformed[key] = value
        return transformed

    return no_rh


def recommended_lambda_sig(variant: str, cfg: dict) -> float:
    if variant == "samwm_no_sigreg":
        return 0.0
    return float(cfg["lambda_sig"])
