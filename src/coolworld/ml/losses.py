from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import WorldModelOutput


@dataclass(frozen=True, slots=True)
class LossTerms:
    total: Tensor
    latent: Tensor
    temperature_nll: Tensor
    latent_variance: Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def world_model_loss(
    output: WorldModelOutput,
    future_temperature: Tensor,
    future_mask: Tensor,
    *,
    latent_weight: float = 1.0,
    temperature_weight: float = 1.0,
    variance_weight: float = 0.05,
    variance_floor: float = 0.5,
) -> LossTerms:
    if output.latent_target is None:
        raise ValueError("training loss requires future latent targets")

    mask4 = future_mask.unsqueeze(-1)
    latent_error = (output.latent_pred - output.latent_target).pow(2).mean(dim=-1)
    latent_loss = _masked_mean(latent_error, future_mask)

    log_scale = output.temperature_log_scale
    inv_var = torch.exp(-2.0 * log_scale)
    nll = 0.5 * (future_temperature - output.temperature_mean).pow(2) * inv_var + log_scale
    temperature_nll = _masked_mean(nll, future_mask)

    # Collapse guard: valid latent dimensions should retain non-trivial variation.
    valid_latent = output.latent_pred[mask4.expand_as(output.latent_pred)].view(
        -1, output.latent_pred.shape[-1]
    )
    if valid_latent.shape[0] > 1:
        std = torch.sqrt(valid_latent.var(dim=0, unbiased=False) + 1e-4)
        latent_variance = torch.relu(variance_floor - std).mean()
    else:
        latent_variance = output.latent_pred.new_zeros(())

    total = (
        latent_weight * latent_loss
        + temperature_weight * temperature_nll
        + variance_weight * latent_variance
    )
    return LossTerms(
        total=total,
        latent=latent_loss,
        temperature_nll=temperature_nll,
        latent_variance=latent_variance,
    )
