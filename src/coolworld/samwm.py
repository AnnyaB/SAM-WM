from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SAMWMOutput:
    temperature_mean: Tensor
    temperature_log_scale: Tensor
    latent_pred: Tensor
    latent_target: Tensor | None
    surprise: Tensor | None
    exchange_conservation_error: Tensor
    mechanism_weights: Tensor


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian regularizer adapted from LeWM's public implementation."""

    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: Tensor) -> Tensor:
        # z: [B,T,N,D] or [M,D]
        z = z.reshape(-1, z.shape[-1])
        if z.shape[0] < 2:
            return z.new_zeros(())
        a = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-8)
        x_t = (z @ a).unsqueeze(-1) * self.t.to(z.dtype)
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        stat = (err @ self.weights.to(z.dtype)) * z.shape[0]
        return stat.mean()


class SparseAdaptiveMentalMap(nn.Module):
    """O(E) sparse message passing over a precomputed undirected city graph."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, z: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # z [B,N,D], edge_index [2,E] with undirected pair list i<j
        src, dst = edge_index
        zi, zj = z[:, src], z[:, dst]
        ea = edge_attr.unsqueeze(0).expand(z.shape[0], -1, -1)
        m_ij = self.msg(torch.cat([zi, zj, ea], dim=-1))
        m_ji = self.msg(torch.cat([zj, zi, ea], dim=-1))
        agg = z.new_zeros(z.shape)
        agg.index_add_(1, dst, m_ij)
        agg.index_add_(1, src, m_ji)
        return self.upd(agg.reshape(-1, z.shape[-1]), z.reshape(-1, z.shape[-1])).view_as(z)


class ConservativeExchange(nn.Module):
    """Pairwise antisymmetric heat exchange; the sum of node increments is zero."""

    def __init__(self, hidden_dim: int, max_fraction: float = 0.45) -> None:
        super().__init__()
        self.max_fraction = float(max_fraction)
        self.kappa = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        temp: Tensor,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        zi, zj = z[:, src], z[:, dst]
        ea = edge_attr.unsqueeze(0).expand(z.shape[0], -1, -1)
        symmetric = torch.cat([zi + zj, (zi - zj).abs(), ea], dim=-1)
        # Positive pair conductance, then degree-normalise it so each node's
        # total exchange coefficient is <= max_fraction. This keeps one exchange
        # step a convex mixing operator (discrete maximum principle) while
        # preserving exact antisymmetry/conservation.
        raw = F.softplus(self.kappa(symmetric).squeeze(-1)) * edge_weight
        load = temp.new_zeros(temp.shape)
        load.index_add_(1, src, raw)
        load.index_add_(1, dst, raw)
        node_scale = torch.clamp(self.max_fraction / load.clamp_min(1e-8), max=1.0)
        edge_scale = torch.minimum(node_scale[:, src], node_scale[:, dst])
        k = raw * edge_scale
        flux = k * (temp[:, dst] - temp[:, src])
        delta = temp.new_zeros(temp.shape)
        delta.index_add_(1, src, flux)
        delta.index_add_(1, dst, -flux)
        conservation_error = delta.sum(dim=1).abs().mean()
        return delta, conservation_error


class WindTransport(nn.Module):
    """Conservative upwind transport. It becomes identically zero when wind is unavailable."""

    def __init__(self, max_fraction: float = 0.25) -> None:
        super().__init__()
        self.max_fraction = float(max_fraction)
        self.raw_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        temp: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        wind_uv: Tensor | None,
        edge_weight: Tensor,
    ) -> Tensor:
        if wind_uv is None:
            return torch.zeros_like(temp)
        src, dst = edge_index
        # edge_attr[:,0:2] = local unit direction x,y from src->dst
        direction = edge_attr[:, :2]
        w = 0.5 * (wind_uv[:, src] + wind_uv[:, dst])
        speed_along = (w * direction.unsqueeze(0)).sum(-1)
        q = self.max_fraction * torch.sigmoid(self.raw_scale) * torch.tanh(speed_along)
        q = q * edge_weight
        upwind = torch.where(q >= 0, temp[:, src], temp[:, dst])
        flux = q * upwind
        delta = temp.new_zeros(temp.shape)
        delta.index_add_(1, src, -flux)
        delta.index_add_(1, dst, flux)
        return delta


class SAMWorldModel(nn.Module):
    """Sparse Adaptive Mechanism World Model.

    The model learns a compact latent state, routes a small typed mechanism library,
    dreams future states autoregressively, and keeps physically constrained structure
    in the operators rather than in a large collection of penalty terms.
    """

    def __init__(
        self,
        dynamic_dim: int = 2,
        static_dim: int = 3,
        hidden_dim: int = 64,
        max_source_step_c: float = 2.0,
        residual_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.max_source_step_c = float(max_source_step_c)
        self.residual_fraction = float(residual_fraction)

        self.frame_encoder = nn.Sequential(
            nn.Linear(dynamic_dim + static_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mental_map = SparseAdaptiveMentalMap(hidden_dim)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4)
        )
        self.exchange = ConservativeExchange(hidden_dim)
        self.transport = WindTransport()
        self.source = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.latent_dynamics = nn.GRUCell(hidden_dim + 5, hidden_dim)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )

    @staticmethod
    def _time_features(time_hours: Tensor) -> Tensor:
        # time_hours [B,T] or [B,H], UTC hour index since epoch-like origin
        day = 24.0
        year = 24.0 * 365.2425
        return torch.stack(
            [
                torch.sin(2 * torch.pi * time_hours / day),
                torch.cos(2 * torch.pi * time_hours / day),
                torch.sin(2 * torch.pi * time_hours / year),
                torch.cos(2 * torch.pi * time_hours / year),
            ],
            dim=-1,
        )

    def _encode_frames(self, dynamic: Tensor, static: Tensor, time_hours: Tensor) -> Tensor:
        # dynamic [B,T,N,F], static [B,N,S]
        b, t, n, _ = dynamic.shape
        tf = self._time_features(time_hours).unsqueeze(2).expand(-1, -1, n, -1)
        st = static.unsqueeze(1).expand(-1, t, -1, -1)
        x = torch.cat([dynamic, st, tf], dim=-1)
        return self.frame_encoder(x)

    def forward(
        self,
        context_dynamic: Tensor,
        static: Tensor,
        context_time_hours: Tensor,
        future_time_hours: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        *,
        context_temperature_index: int = 0,
        future_dynamic_target: Tensor | None = None,
        future_temperature_target: Tensor | None = None,
        future_wind_uv: Tensor | None = None,
    ) -> SAMWMOutput:
        b, t, n, _ = context_dynamic.shape
        h = future_time_hours.shape[1]
        enc = self._encode_frames(context_dynamic, static, context_time_hours)
        seq = enc.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden_dim)
        _, h_n = self.temporal(seq)
        z = h_n[-1].view(b, n, self.hidden_dim)
        z = self.mental_map(z, edge_index, edge_attr)
        temp = context_dynamic[:, -1, :, context_temperature_index]

        means, scales, latents, weights = [], [], [], []
        conservation = []
        for step in range(h):
            tf = self._time_features(future_time_hours[:, step]).unsqueeze(1).expand(-1, n, -1)
            logits = self.router(torch.cat([z, tf], dim=-1))
            # The transport mechanism is unavailable unless observed wind is
            # explicitly supplied for the rollout. Masking it before softmax
            # prevents the router from assigning probability mass to a branch
            # that is physically unsupported in the current dataset.
            if future_wind_uv is None:
                logits = logits.clone()
                logits[..., 1] = torch.finfo(logits.dtype).min
            route = torch.softmax(logits, dim=-1)
            src, dst = edge_index
            edge_route_exchange = 0.5 * (route[:, src, 0] + route[:, dst, 0])
            edge_route_transport = 0.5 * (route[:, src, 1] + route[:, dst, 1])
            ex, cons = self.exchange(temp, z, edge_index, edge_attr, edge_route_exchange)
            wind = None if future_wind_uv is None else future_wind_uv[:, step]
            tr = self.transport(temp, edge_index, edge_attr, wind, edge_route_transport)
            src_term = self.max_source_step_c * torch.tanh(
                self.source(torch.cat([z, tf], -1)).squeeze(-1)
            )
            src_term = src_term * route[:, :, 2]
            res = (
                self.residual_fraction
                * self.max_source_step_c
                * torch.tanh(self.residual(torch.cat([z, tf], -1)).squeeze(-1))
            )
            res = res * route[:, :, 3]
            delta = ex + tr + src_term + res
            temp = temp + delta
            dz = torch.cat([z, tf, delta.unsqueeze(-1)], dim=-1)
            z = self.latent_dynamics(
                dz.reshape(-1, self.hidden_dim + 5), z.reshape(-1, self.hidden_dim)
            ).view_as(z)
            z = self.mental_map(z, edge_index, edge_attr)
            means.append(temp)
            scales.append(self.scale_head(z).squeeze(-1).clamp(-4.0, 3.0))
            latents.append(z)
            weights.append(route)
            conservation.append(cons)

        mean = torch.stack(means, dim=1)
        log_scale = torch.stack(scales, dim=1)
        latent_pred = torch.stack(latents, dim=1)
        mechanism_weights = torch.stack(weights, dim=1)
        latent_target = None
        surprise = None
        if future_dynamic_target is not None:
            target_latent = self._encode_frames(future_dynamic_target, static, future_time_hours)
            latent_target = target_latent.detach()
        if future_temperature_target is not None:
            scale = log_scale.exp().clamp_min(1e-4)
            surprise = (future_temperature_target - mean).abs() / scale + log_scale
        return SAMWMOutput(
            temperature_mean=mean,
            temperature_log_scale=log_scale,
            latent_pred=latent_pred,
            latent_target=latent_target,
            surprise=surprise,
            exchange_conservation_error=torch.stack(conservation).mean(),
            mechanism_weights=mechanism_weights,
        )


def samwm_loss(
    output: SAMWMOutput,
    target_temperature: Tensor,
    target_mask: Tensor,
    sigreg: SIGReg,
    lambda_sig: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if output.latent_target is None:
        raise ValueError("latent targets required for training")
    mask = target_mask.to(target_temperature.dtype)
    denom = mask.sum().clamp_min(1.0)
    latent_err = (output.latent_pred - output.latent_target).pow(2).mean(-1)
    pred_latent = (latent_err * mask).sum() / denom
    scale = output.temperature_log_scale.exp().clamp_min(1e-4)
    laplace_nll = (
        target_temperature - output.temperature_mean
    ).abs() / scale + output.temperature_log_scale
    temp_nll = (laplace_nll * mask).sum() / denom
    pred_loss = pred_latent + temp_nll
    # SIGReg must act on a tensor connected to the trainable graph. The JEPA
    # target is deliberately detached for the prediction target, so applying
    # SIGReg to latent_target would contribute zero gradient. Regularise the
    # predicted latent states instead.
    sig = sigreg(output.latent_pred[target_mask]) if target_mask.any() else pred_loss.new_zeros(())
    loss = pred_loss + float(lambda_sig) * sig
    return loss, {
        "loss": loss.detach(),
        "pred_loss": pred_loss.detach(),
        "latent": pred_latent.detach(),
        "temperature_nll": temp_nll.detach(),
        "sigreg": sig.detach(),
    }
