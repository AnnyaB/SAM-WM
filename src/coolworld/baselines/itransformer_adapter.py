from __future__ import annotations

from torch import Tensor, nn


class ITransformerAdapter(nn.Module):
    """Matched node-token baseline inspired by iTransformer (ICLR 2024).

    This is an independent task adapter, not the authors' official implementation.
    Each urban sensor is represented as one token containing its complete dynamic
    lookback. Self-attention mixes information across sensors while shared weights
    allow evaluation on cities with a different number of sensors.
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
        b, t, n, f = context_dynamic.shape
        if t != self.context_hours or f != self.dynamic_dim:
            raise ValueError(
                f"expected context [B,{self.context_hours},N,{self.dynamic_dim}], "
                f"received {tuple(context_dynamic.shape)}"
            )
        tokens = context_dynamic.permute(0, 2, 1, 3).reshape(b, n, t * f)
        z = self.token_proj(tokens) + self.static_proj(static)
        z = self.norm(self.encoder(z))
        return self.head(z).transpose(1, 2)
