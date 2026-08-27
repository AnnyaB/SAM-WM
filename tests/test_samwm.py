import torch

from coolworld.samwm import SAMWorldModel, SIGReg, samwm_loss


def fixture():
    batch, context, nodes, horizon = 2, 8, 6, 3
    x = torch.randn(batch, context, nodes, 3)
    static = torch.randn(batch, nodes, 3)
    context_time = torch.arange(context).float().repeat(batch, 1)
    future_time = torch.arange(context, context + horizon).float().repeat(batch, 1)
    pairs = [(i, i + 1) for i in range(nodes - 1)]
    edge_index = torch.tensor(pairs).t().long()
    edge_attr = torch.tensor([[1.0, 0.0, 0.1]] * len(pairs))
    future = torch.randn(batch, horizon, nodes, 3)
    temp = torch.randn(batch, horizon, nodes)
    mask = torch.ones(batch, horizon, nodes, dtype=torch.bool)
    return x, static, context_time, future_time, edge_index, edge_attr, future, temp, mask


def test_forward_and_backward():
    x, static, ct, ft, edge_index, edge_attr, future, temp, mask = fixture()
    model = SAMWorldModel(hidden_dim=32)
    output = model(
        x,
        static,
        ct,
        ft,
        edge_index,
        edge_attr,
        future_dynamic_target=future,
        future_temperature_target=temp,
    )
    assert output.temperature_mean.shape == temp.shape
    loss, _ = samwm_loss(output, temp, mask, SIGReg(num_proj=16), 0.01)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_exchange_conserves_pair_flux():
    x, static, ct, ft, edge_index, edge_attr, _, _, _ = fixture()
    model = SAMWorldModel(hidden_dim=32)
    output = model(x, static, ct, ft, edge_index, edge_attr)
    assert float(output.exchange_conservation_error.detach()) < 1e-5


def test_missing_wind_masks_transport_router():
    x, static, ct, ft, edge_index, edge_attr, _, _, _ = fixture()
    model = SAMWorldModel(hidden_dim=32)
    output = model(x, static, ct, ft, edge_index, edge_attr)
    assert torch.isfinite(output.temperature_mean).all()
    assert torch.all(output.mechanism_weights[..., 1] < 1e-7)


def test_sigreg_has_trainable_gradient():
    x, static, ct, ft, edge_index, edge_attr, future, _, mask = fixture()
    model = SAMWorldModel(hidden_dim=32)
    output = model(x, static, ct, ft, edge_index, edge_attr, future_dynamic_target=future)
    regularizer = SIGReg(num_proj=16)(output.latent_pred[mask])
    regularizer.backward()
    assert model.latent_dynamics.weight_hh.grad is not None
    assert torch.isfinite(model.latent_dynamics.weight_hh.grad).all()


def test_exchange_obeys_global_maximum_principle():
    x, static, ct, _, edge_index, edge_attr, _, _, _ = fixture()
    model = SAMWorldModel(hidden_dim=32)
    with torch.no_grad():
        enc = model._encode_frames(x, static, ct)
        batch, context, nodes, _ = x.shape
        seq = enc.permute(0, 2, 1, 3).reshape(batch * nodes, context, model.hidden_dim)
        _, hidden = model.temporal(seq)
        latent = hidden[-1].view(batch, nodes, model.hidden_dim)
        temp = x[:, -1, :, 0]
        weight = torch.ones(batch, edge_index.shape[1])
        delta, _ = model.exchange(temp, latent, edge_index, edge_attr, weight)
        next_temp = temp + delta
        assert (next_temp >= temp.min(dim=1, keepdim=True).values - 1e-6).all()
        assert (next_temp <= temp.max(dim=1, keepdim=True).values + 1e-6).all()
