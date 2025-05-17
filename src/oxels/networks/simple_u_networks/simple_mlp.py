import torch
import torch.nn as nn

from oxels.typing import ActivationType
from .simple_norm import SimpleNorm


class SimpleMLPBlock(nn.Module):
    """
    Minimal MLP block (no time-conditioning):

    This block performs the following steps on an input x of shape (B, C, H, W):
      1) x_norm = Normalize(x) using SimpleNorm(center=False)
      2) h = Dense(x_norm) → C * expansion
      3) h = activation_fn(h) + Dropout
      4) out = Dense(h) → C, with zero-initialized weights

    Comparison to pseudocode:

      def mlp_block(x, emb, expansion_factor=4):
          B, HW, C = x.shape
          x = Normalize(x)
          mlp_h = Dense(x, expansion_factor * C)
          scale = DenseGeneral(emb, mlp_h.shape[2:])
          shift = DenseGeneral(emb, mlp_h.shape[2:])
          mlp_h = swish(mlp_h)
          mlp_h = mlp_h * (1. + scale[:, None]) + shift[:, None]  # if dropout > 0
          mlp_h = Dropout(mlp_h, config.transformer_dropout)
          out = Dense(mlp_h, C, kernel_init=zeros)
          return out

      - We remove `emb` and scale/shift injection (no time-conditioning).
      - Normalize(x) here uses SimpleNorm(method='layer', center=False, scale=True).
      - Activation uses F.silu (Swish).
      - Dropout is applied after activation.
      - Final projection `dense_down` uses zero initialization to match `kernel_init=zeros`.

    Note on normalization:
      - `Normalize` (center=False) applies only scaling.
      - `NormalizeWithBias` (center=True) also adds a learnable bias/shift.
      In attention blocks, we use both forms accordingly.
    """

    def __init__(
        self, channel_dim: int, expansion: int = 4, dropout: float = 0.0, activation: ActivationType = nn.SiLU
    ):
        super().__init__()
        # Normalize without bias (scale only)
        self.norm = SimpleNorm(channel_dim, method="layer", center=False, scale=True)
        self.dense_up = nn.Linear(channel_dim, channel_dim * expansion)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.dense_down = nn.Linear(channel_dim * expansion, channel_dim)
        nn.init.normal_(self.dense_down.weight, mean=0.0, std=1-6)
        nn.init.normal_(self.dense_down.bias, mean=0.0, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.norm(x)
        # move channels to last dim for Linear
        h = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        h = self.dense_up(h)  # (B, H, W, C*expansion)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.dense_down(h)  # (B, H, W, C)
        # back to (B, C, H, W)
        return h.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)


if __name__ == "__main__":
    # Simple executable test
    batch, channels, height, width = 2, 16, 32, 32
    x = torch.randn(batch, channels, height, width)
    block = SimpleMLPBlock(channel_dim=channels, expansion=4, dropout=0.1)
    y = block(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
