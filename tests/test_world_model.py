import pytest

torch = pytest.importorskip("torch")
model_module = pytest.importorskip("coolworld.ml.model")
ActionConditionedJEPAWorldModel = model_module.ActionConditionedJEPAWorldModel


def test_world_model_shapes_and_future_conditioning() -> None:
    b, t, h, n = 2, 4, 3, 5
    model = ActionConditionedJEPAWorldModel(
        3, 2, 3, 2, latent_dim=32, spatial_layers=1, spatial_heads=4, dropout=0.0
    )
    context_dynamic = torch.randn(b, t, n, 3)
    context_actions = torch.zeros(b, t, n, 3)
    context_mask = torch.ones(b, t, n, dtype=torch.bool)
    static = torch.randn(b, n, 2)
    future_actions = torch.zeros(b, h, n, 3)
    future_known = torch.randn(b, h, n, 2)
    future_mask = torch.ones(b, h, n, dtype=torch.bool)
    future_dynamic = torch.randn(b, h, n, 3)
    out = model(
        context_dynamic,
        context_actions,
        context_mask,
        static,
        future_actions,
        future_known,
        future_mask=future_mask,
        future_dynamic=future_dynamic,
    )
    assert out.temperature_mean.shape == (b, h, n)
    assert out.temperature_log_scale.shape == (b, h, n)
    assert out.latent_pred.shape == (b, h, n, 32)
    assert out.latent_target is not None and out.latent_target.shape == (b, h, n, 32)
