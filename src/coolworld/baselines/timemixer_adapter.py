from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _TemporalMixer(nn.Module):
    def __init__(self, length: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(length, hidden),
            nn.GELU(),
            nn.Linear(hidden, length),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class TimeMixerAdapter(nn.Module):
    """multiscale baseline inspired by TimeMixer (ICLR 2024).

    This is an independent task adapter, not the authors' official implementation.
    It retains the central multiscale seasonal/trend mixing idea while using shared
    per-sensor weights so the model can be evaluated zero-shot on a city with a
    different number of sensors.
    """

    def __init__(
        self,
        *,
        context_hours: int,
        horizon_hours: int,
        dynamic_dim: int = 3,
        d_model: int = 64,
        scales: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if any(context_hours % scale for scale in scales):
            raise ValueError("all TimeMixer scales must divide context_hours")
        self.context_hours = int(context_hours)
        self.horizon_hours = int(horizon_hours)
        self.dynamic_dim = int(dynamic_dim)
        self.scales = tuple(map(int, scales))
        self.input_proj = nn.Linear(dynamic_dim, d_model)
        self.season_mix = nn.ModuleDict()
        self.trend_mix = nn.ModuleDict()
        for scale in self.scales:
            length = context_hours // scale
            hidden = max(2 * length, 16)
            self.season_mix[str(scale)] = _TemporalMixer(length, hidden)
            self.trend_mix[str(scale)] = _TemporalMixer(length, hidden)
        self.feature_mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.temporal_head = nn.Linear(context_hours, horizon_hours)
        self.output_head = nn.Linear(d_model, 1)

    @staticmethod
    def _decompose(x: Tensor) -> tuple[Tensor, Tensor]:
        trend = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        return x - trend, trend

    def forward(self, context_dynamic: Tensor) -> Tensor:
        b, t, n, f = context_dynamic.shape
        if t != self.context_hours or f != self.dynamic_dim:
            raise ValueError(
                f"expected context [B,{self.context_hours},N,{self.dynamic_dim}], "
                f"received {tuple(context_dynamic.shape)}"
            )
        x = self.input_proj(context_dynamic).permute(0, 2, 1, 3).reshape(b * n, t, -1)
        x = x.transpose(1, 2)
        mixed: list[Tensor] = []
        for scale in self.scales:
            xs = x if scale == 1 else F.avg_pool1d(x, kernel_size=scale, stride=scale)
            season, trend = self._decompose(xs)
            ys = self.season_mix[str(scale)](season) + self.trend_mix[str(scale)](trend)
            if scale != 1:
                ys = F.interpolate(ys, size=t, mode="linear", align_corners=False)
            mixed.append(ys)
        z = torch.stack(mixed, dim=0).mean(0).transpose(1, 2)
        z = z + self.feature_mix(z)
        horizon = self.temporal_head(z.transpose(1, 2)).transpose(1, 2)
        pred = self.output_head(horizon).squeeze(-1).view(b, n, self.horizon_hours)
        return pred.transpose(1, 2)
