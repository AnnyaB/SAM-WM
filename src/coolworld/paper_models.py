from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .samwm import SAMWorldModel


class ITransformerAdapter(nn.Module):
    """Node-token inverted Transformer for the SAM-WM benchmark protocol.

    This is an independent task adapter inspired by the variate-token construction in
    iTransformer (Liu et al., ICLR 2024 Spotlight). It is not copied from the authors'
    source code and must not be described as their official implementation.

    Each urban sensor is one token. The token contains its complete dynamic lookback;
    self-attention therefore mixes information across sensors while the same weights
    work for cities with different sensor counts.
    """

    def __init__(
        self,
        *,
        context_hours: int,
        horizon_hours: int,
        dynamic_dim: int = 3,
        static_dim: int = 3,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.context_hours = int(context_hours)
        self.horizon_hours = int(horizon_hours)
        self.dynamic_dim = int(dynamic_dim)
        self.token_proj = nn.Linear(context_hours * dynamic_dim, d_model)
        self.static_proj = nn.Linear(static_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon_hours)

    def forward(self, context_dynamic: Tensor, static: Tensor) -> Tensor:
        # [B,T,N,F] -> [B,N,T*F]
        b, t, n, f = context_dynamic.shape
        if t != self.context_hours or f != self.dynamic_dim:
            raise ValueError(
                f"expected context [B,{self.context_hours},N,{self.dynamic_dim}], "
                f"received {tuple(context_dynamic.shape)}"
            )
        tokens = context_dynamic.permute(0, 2, 1, 3).reshape(b, n, t * f)
        z = self.token_proj(tokens) + self.static_proj(static)
        z = self.norm(self.encoder(z))
        # [B,N,H] -> [B,H,N]
        return self.head(z).transpose(1, 2)


class _TemporalMixer(nn.Module):
    def __init__(self, length: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(length, hidden),
            nn.GELU(),
            nn.Linear(hidden, length),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,D,T]; mix only along the temporal axis.
        return x + self.net(x)


class TimeMixerAdapter(nn.Module):
    """Compact multi-scale temporal-mixing baseline for the urban benchmark.

    This independent implementation follows the central TimeMixer idea of decomposable
    multi-scale temporal mixing (Wang et al., ICLR 2024). It uses shared per-sensor
    weights (channel-independent transfer) so it remains evaluable zero-shot when the
    target city has a different number of sensors. It is not the authors' official code.
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
        # x: [BN,D,L]
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
        x = x.transpose(1, 2)  # [BN,D,T]
        mixed: list[Tensor] = []
        for scale in self.scales:
            if scale == 1:
                xs = x
            else:
                xs = F.avg_pool1d(x, kernel_size=scale, stride=scale)
            season, trend = self._decompose(xs)
            ys = self.season_mix[str(scale)](season) + self.trend_mix[str(scale)](trend)
            if scale != 1:
                ys = F.interpolate(ys, size=t, mode="linear", align_corners=False)
            mixed.append(ys)
        z = torch.stack(mixed, dim=0).mean(0).transpose(1, 2)  # [BN,T,D]
        z = z + self.feature_mix(z)
        horizon = self.temporal_head(z.transpose(1, 2)).transpose(1, 2)  # [BN,H,D]
        pred = self.output_head(horizon).squeeze(-1).view(b, n, self.horizon_hours)
        return pred.transpose(1, 2)


class IdentityMentalMap(nn.Module):
    """Ablation module: remove sparse learned message passing."""

    def forward(self, z: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:  # noqa: ARG002
        return z


class ZeroExchange(nn.Module):
    """Ablation module: remove the conservative pair-exchange mechanism."""

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

BASELINES = ("itransformer", "timemixer")
PAPER_MODELS = ("samwm", *BASELINES, *SAM_ABLATIONS[1:])


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


def build_baseline(name: str, cfg: dict) -> nn.Module:
    common = {
        "context_hours": int(cfg["context_hours"]),
        "horizon_hours": int(cfg["horizon_hours"]),
        "dynamic_dim": 3,
    }
    if name == "itransformer":
        return ITransformerAdapter(
            **common,
            static_dim=3,
            d_model=int(cfg.get("itransformer_d_model", 96)),
            n_heads=int(cfg.get("itransformer_heads", 4)),
            n_layers=int(cfg.get("itransformer_layers", 3)),
            dropout=float(cfg.get("baseline_dropout", 0.10)),
        )
    if name == "timemixer":
        return TimeMixerAdapter(
            **common,
            d_model=int(cfg.get("timemixer_d_model", 64)),
            dropout=float(cfg.get("baseline_dropout", 0.10)),
        )
    raise ValueError(f"unknown baseline {name!r}")


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


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def recommended_lambda_sig(variant: str, cfg: dict) -> float:
    if variant == "samwm_no_sigreg":
        return 0.0
    return float(cfg["lambda_sig"])


def baseline_display_name(name: str) -> str:
    return {
        "samwm": "SAM-WM",
        "itransformer": "iTransformer-adapted",
        "timemixer": "TimeMixer-adapted",
        "samwm_no_sigreg": "SAM-WM − SIGReg",
        "samwm_no_exchange": "SAM-WM − exchange",
        "samwm_no_mental_map": "SAM-WM − mental map",
        "samwm_no_residual": "SAM-WM − residual",
        "samwm_no_rh": "SAM-WM − RH",
    }[name]


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def standard_error(std: float, n: int) -> float:
    return float(std / math.sqrt(max(n, 1)))
