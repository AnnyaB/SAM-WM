from __future__ import annotations

import research
import torch

from coolworld.samwm import SAMWorldModel, SIGReg


def _toy_inputs():
    context_dynamic = torch.randn(1, 4, 3, 3)
    static = torch.randn(1, 3, 3)
    context_time = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    future_time = torch.arange(4, 6, dtype=torch.float32).unsqueeze(0)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.7]], dtype=torch.float32
    )
    future_dynamic = torch.randn(1, 2, 3, 3)
    future_temperature = torch.randn(1, 2, 3)
    return (
        context_dynamic,
        static,
        context_time,
        future_time,
        edge_index,
        edge_attr,
        future_dynamic,
        future_temperature,
    )


def test_research_contract_is_five_seeds_and_seven_structural_ablations():
    assert research.RESEARCH_SEEDS == (17, 29, 42, 73, 101)
    assert len(research.ABLATIONS) == 7
    assert set(research.CONTROLS) == {"no_sigreg", "temperature_only"}


def test_every_structural_ablation_preserves_forward_contract():
    (
        context_dynamic,
        static,
        context_time,
        future_time,
        edge_index,
        edge_attr,
        future_dynamic,
        future_temperature,
    ) = _toy_inputs()

    for variant in ("full", *research.ABLATIONS):
        model = SAMWorldModel(hidden_dim=8, max_source_step_normalized=1.0)
        model = research.apply_variant(model, variant)
        output = model(
            context_dynamic,
            static,
            context_time,
            future_time,
            edge_index,
            edge_attr,
            future_dynamic_target=future_dynamic,
            future_temperature_target=future_temperature,
        )
        assert output.temperature_mean.shape == (1, 2, 3)
        assert output.temperature_log_scale.shape == (1, 2, 3)
        assert output.latent_pred.shape == (1, 2, 3, 8)
        assert torch.isfinite(output.temperature_mean).all()


def test_temperature_only_control_has_gradient_and_no_latent_prediction_term():
    (
        context_dynamic,
        static,
        context_time,
        future_time,
        edge_index,
        edge_attr,
        future_dynamic,
        future_temperature,
    ) = _toy_inputs()
    model = SAMWorldModel(hidden_dim=8, max_source_step_normalized=1.0)
    output = model(
        context_dynamic,
        static,
        context_time,
        future_time,
        edge_index,
        edge_attr,
        future_dynamic_target=future_dynamic,
        future_temperature_target=future_temperature,
    )
    target_mask = torch.ones_like(future_temperature, dtype=torch.bool)
    loss, parts = research._temperature_only_loss(
        output,
        future_temperature,
        target_mask,
        SIGReg(num_proj=4),
        0.01,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["latent"].item() == 0.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_research_runtime_restores_production_functions():
    original_build = research.experiment._build_model
    original_loss = research.experiment.samwm_loss
    with research.research_runtime("temperature_only"):
        assert research.experiment._build_model is not original_build
        assert research.experiment.samwm_loss is research._temperature_only_loss
    assert research.experiment._build_model is original_build
    assert research.experiment.samwm_loss is original_loss
