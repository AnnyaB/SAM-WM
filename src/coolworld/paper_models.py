from __future__ import annotations

import math

from torch import nn

from .ablations import (
    SAM_ABLATIONS,
    IdentityMentalMap,
    ZeroExchange,
    batch_transform_for_variant,
    build_samwm,
    recommended_lambda_sig,
)
from .baselines import ITransformerAdapter, TimeMixerAdapter

BASELINES = ("itransformer", "timemixer")
PAPER_MODELS = ("samwm", *BASELINES, *SAM_ABLATIONS[1:])


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


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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


__all__ = [
    "BASELINES",
    "PAPER_MODELS",
    "SAM_ABLATIONS",
    "ITransformerAdapter",
    "IdentityMentalMap",
    "TimeMixerAdapter",
    "ZeroExchange",
    "baseline_display_name",
    "batch_transform_for_variant",
    "build_baseline",
    "build_samwm",
    "parameter_count",
    "recommended_lambda_sig",
    "safe_mean",
    "standard_error",
]
