from __future__ import annotations

import torch

from coolworld.paper_models import (
    ITransformerAdapter,
    TimeMixerAdapter,
    build_samwm,
)


def _batch(nodes: int = 5):
    context = torch.randn(2, 8, nodes, 3)
    static = torch.randn(2, nodes, 3)
    return context, static


def test_itransformer_adapter_supports_variable_city_size() -> None:
    model = ITransformerAdapter(
        context_hours=8,
        horizon_hours=3,
        d_model=32,
        n_heads=4,
        n_layers=1,
    )
    for nodes in (5, 9):
        context, static = _batch(nodes)
        assert model(context, static).shape == (2, 3, nodes)


def test_timemixer_adapter_supports_variable_city_size() -> None:
    model = TimeMixerAdapter(context_hours=8, horizon_hours=3, d_model=16)
    for nodes in (5, 9):
        context, _ = _batch(nodes)
        assert model(context).shape == (2, 3, nodes)


def test_samwm_ablation_modules_preserve_output_contract() -> None:
    cfg = {
        "hidden_dim": 16,
        "max_source_step_normalized": 1.0,
        "residual_fraction": 0.2,
    }
    context = torch.randn(2, 8, 5, 3)
    static = torch.randn(2, 5, 3)
    context_time = torch.arange(8, dtype=torch.float32).repeat(2, 1)
    future_time = torch.arange(8, 11, dtype=torch.float32).repeat(2, 1)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 3)
    for variant in ("samwm", "samwm_no_exchange", "samwm_no_mental_map", "samwm_no_residual"):
        model = build_samwm(cfg, variant)
        output = model(
            context,
            static,
            context_time,
            future_time,
            edge_index,
            edge_attr,
        )
        assert output.temperature_mean.shape == (2, 3, 5)
        assert output.temperature_log_scale.shape == (2, 3, 5)
