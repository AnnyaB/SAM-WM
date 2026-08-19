from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class WorldModelOutput:
    latent_pred: Tensor
    latent_target: Tensor | None
    temperature_mean: Tensor
    temperature_log_scale: Tensor


class ActionConditionedJEPAWorldModel(nn.Module):
    """Action-conditioned spatiotemporal predictive world model.

    Observed states are encoded with an online encoder. A target encoder updated
    by EMA supplies stop-gradient future latent targets. Each context frame uses
    a spatial Transformer across real city tiles; temporal state is aggregated
    per tile with a GRU. Future rollouts are conditioned on both the physical
    action vector and *known future exogenous features* (currently time-of-day),
    and spatial interaction is reapplied at every rollout step.

    The model never receives future target temperature as a rollout input.
    """

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        action_dim: int,
        known_future_dim: int,
        *,
        latent_dim: int = 128,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.1,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        if latent_dim % spatial_heads != 0:
            raise ValueError("latent_dim must be divisible by spatial_heads")
        if known_future_dim <= 0:
            raise ValueError("known_future_dim must be positive")
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.action_dim = action_dim
        self.known_future_dim = known_future_dim
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        self.state_encoder = nn.Sequential(
            nn.Linear(dynamic_dim + static_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.target_state_encoder = deepcopy(self.state_encoder)
        for p in self.target_state_encoder.parameters():
            p.requires_grad_(False)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )
        self.future_condition_encoder = nn.Sequential(
            nn.Linear(action_dim + known_future_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=spatial_heads,
            dim_feedforward=4 * latent_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(layer, num_layers=spatial_layers)
        self.temporal = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.rollout_cell = nn.GRUCell(latent_dim, latent_dim)
        self.latent_predictor = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.temperature_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 2),
        )

    def _encode_states(self, dynamic: Tensor, static: Tensor, *, target: bool) -> Tensor:
        b, t, n, _ = dynamic.shape
        static_t = static[:, None].expand(b, t, n, self.static_dim)
        encoder = self.target_state_encoder if target else self.state_encoder
        return encoder(torch.cat([dynamic, static_t], dim=-1))

    def _spatial_context(self, latent: Tensor, action: Tensor, mask: Tensor) -> Tensor:
        b, t, n, d = latent.shape
        z = latent + self.action_encoder(action)
        flat = z.reshape(b * t, n, d)
        padding = (~mask.bool()).reshape(b * t, n)
        flat = self.spatial_encoder(flat, src_key_padding_mask=padding)
        return flat.reshape(b, t, n, d)

    def _temporal_context(self, spatial: Tensor) -> Tensor:
        b, t, n, d = spatial.shape
        per_tile = spatial.permute(0, 2, 1, 3).reshape(b * n, t, d)
        _, h = self.temporal(per_tile)
        return h[-1].reshape(b, n, d)

    def forward(
        self,
        context_dynamic: Tensor,
        context_actions: Tensor,
        context_mask: Tensor,
        static: Tensor,
        future_actions: Tensor,
        future_known: Tensor,
        *,
        future_mask: Tensor | None = None,
        future_dynamic: Tensor | None = None,
    ) -> WorldModelOutput:
        if context_dynamic.ndim != 4 or future_actions.ndim != 4 or future_known.ndim != 4:
            raise ValueError("world model expects [B,T,N,F] tensors")
        if future_actions.shape[:3] != future_known.shape[:3]:
            raise ValueError("future actions/known features must align")
        if future_known.shape[-1] != self.known_future_dim:
            raise ValueError("future_known feature dimension mismatch")

        context_latent = self._encode_states(context_dynamic, static, target=False)
        spatial = self._spatial_context(context_latent, context_actions, context_mask)
        hidden = self._temporal_context(spatial)

        b, h, n, _ = future_actions.shape
        if future_mask is None:
            future_mask = context_mask[:, -1:, :].expand(b, h, n)
        if future_mask.shape != (b, h, n):
            raise ValueError("future_mask must have shape [B,H,N]")

        state = hidden.reshape(b * n, self.latent_dim)
        latent_steps: list[Tensor] = []
        mean_steps: list[Tensor] = []
        scale_steps: list[Tensor] = []
        for step in range(h):
            cond = torch.cat([future_actions[:, step], future_known[:, step]], dim=-1)
            cond_z = self.future_condition_encoder(cond)
            state = self.rollout_cell(cond_z.reshape(b * n, self.latent_dim), state)

            # Re-couple tiles at every imagined step so local actions may affect
            # nearby thermal state through learned spatial dynamics.
            state_grid = state.reshape(b, n, self.latent_dim)
            state_grid = self.spatial_encoder(
                state_grid + self.action_encoder(future_actions[:, step]),
                src_key_padding_mask=~future_mask[:, step].bool(),
            )
            state = state_grid.reshape(b * n, self.latent_dim)
            pred = self.latent_predictor(state).reshape(b, n, self.latent_dim)
            params = self.temperature_head(pred)
            latent_steps.append(pred)
            mean_steps.append(params[..., 0])
            scale_steps.append(params[..., 1].clamp(-7.0, 5.0))

        latent_pred = torch.stack(latent_steps, dim=1)
        temperature_mean = torch.stack(mean_steps, dim=1)
        temperature_log_scale = torch.stack(scale_steps, dim=1)

        target_latent: Tensor | None = None
        if future_dynamic is not None:
            with torch.no_grad():
                target_latent = self._encode_states(future_dynamic, static, target=True)

        return WorldModelOutput(latent_pred, target_latent, temperature_mean, temperature_log_scale)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        online = dict(self.state_encoder.named_parameters())
        for name, target_param in self.target_state_encoder.named_parameters():
            target_param.mul_(self.ema_decay).add_(online[name], alpha=1.0 - self.ema_decay)
