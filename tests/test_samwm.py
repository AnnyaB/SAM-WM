import torch

from coolworld.samwm import SAMWorldModel, SIGReg, samwm_loss


def fixture():
    b, t, n, h = 2, 8, 6, 3
    x = torch.randn(b, t, n, 2)
    static = torch.randn(b, n, 3)
    ct = torch.arange(t).float().repeat(b, 1)
    ft = torch.arange(t, t + h).float().repeat(b, 1)
    pairs = []
    for i in range(n - 1):
        pairs.append((i, i + 1))
    ei = torch.tensor(pairs).t().long()
    ea = torch.tensor([[1.0, 0.0]] * len(pairs))
    fut = torch.randn(b, h, n, 2)
    temp = torch.randn(b, h, n)
    mask = torch.ones(b, h, n, dtype=torch.bool)
    return x, static, ct, ft, ei, ea, fut, temp, mask


def test_forward_and_backward():
    x, s, ct, ft, ei, ea, fut, temp, mask = fixture()
    m = SAMWorldModel(hidden_dim=32)
    out = m(x, s, ct, ft, ei, ea, future_dynamic_target=fut, future_temperature_target=temp)
    assert out.temperature_mean.shape == temp.shape
    loss, _ = samwm_loss(out, temp, mask, SIGReg(num_proj=16), 0.01)
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_exchange_conserves_pair_flux():
    x, s, ct, ft, ei, ea, _, _, _ = fixture()
    m = SAMWorldModel(hidden_dim=32)
    out = m(x, s, ct, ft, ei, ea)
    assert float(out.exchange_conservation_error.detach()) < 1e-5


def test_missing_wind_is_valid():
    x, s, ct, ft, ei, ea, _, _, _ = fixture()
    m = SAMWorldModel(hidden_dim=32)
    a = m(x, s, ct, ft, ei, ea).temperature_mean
    assert torch.isfinite(a).all()


def test_sigreg_has_trainable_gradient():
    x, s, ct, ft, ei, ea, fut, temp, mask = fixture()
    m = SAMWorldModel(hidden_dim=32)
    out = m(x, s, ct, ft, ei, ea, future_dynamic_target=fut)
    reg = SIGReg(num_proj=16)(out.latent_pred[mask])
    reg.backward()
    assert m.latent_dynamics.weight_hh.grad is not None
    assert torch.isfinite(m.latent_dynamics.weight_hh.grad).all()


def test_exchange_obeys_global_maximum_principle():
    x, s, ct, ft, ei, ea, _, _, _ = fixture()
    m = SAMWorldModel(hidden_dim=32)
    with torch.no_grad():
        enc = m._encode_frames(x, s, ct)
        b, t, n, _ = x.shape
        seq = enc.permute(0, 2, 1, 3).reshape(b * n, t, m.hidden_dim)
        _, hn = m.temporal(seq)
        z = hn[-1].view(b, n, m.hidden_dim)
        temp = x[:, -1, :, 0]
        weight = torch.ones(b, ei.shape[1])
        delta, _ = m.exchange(temp, z, ei, ea, weight)
        nxt = temp + delta
        assert (nxt >= temp.min(dim=1, keepdim=True).values - 1e-6).all()
        assert (nxt <= temp.max(dim=1, keepdim=True).values + 1e-6).all()


def test_transport_router_is_masked_without_wind():
    x, s, ct, ft, ei, ea, _, _, _ = fixture()
    m = SAMWorldModel(hidden_dim=32)
    out = m(x, s, ct, ft, ei, ea)
    assert torch.all(out.mechanism_weights[..., 1] < 1e-7)
