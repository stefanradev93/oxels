import math
import torch
import torch.nn as nn

from oxels.types import ActivationType
from simple_norm import SimpleNorm


class SimpleResidualBlock(nn.Module):
    """
    Residual block without time embeddings:

    Applies a two-convolution residual block with optional skip connection:
      h = NormalizeWithBias(x)
      if skip_h is not None:
          skip_h = NormalizeWithBias(skip_h)
          h = (h + skip_h) / sqrt(2)

      h = activation(h)
      h = Conv2d(h, out_ch)

      h = NormalizeWithBias(h)
      h = activation(h)
      h = Dropout(h)
      h = Conv2d_zero_init(h, out_ch)

      return x + h
    """
    def __init__(self,
                 out_channels: int,
                 activation: ActivationType=nn.SiLU,
                 has_skip: bool = False,
                 dropout: float = 0.0):
        super(SimpleResidualBlock, self).__init__()
        self.out_channels = out_channels
        self.has_skip = has_skip
        self.activation = activation()

        # NormalizeWithBias for input
        self.norm_in = SimpleNorm(
            channel_dim=out_channels,
            method='layer',
            center=True,
            scale=True
        )
        # Optional skip normalization with bias
        if has_skip:
            self.norm_skip = SimpleNorm(
                channel_dim=out_channels,
                method='layer',
                center=True,
                scale=True
            )
        # First convolution: Xavier uniform init
        self.conv_in = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        nn.init.xavier_uniform_(self.conv_in.weight, gain=1.0)
        nn.init.zeros_(self.conv_in.bias)

        # NormalizeWithBias after first conv
        self.norm_out = SimpleNorm(
            channel_dim=out_channels,
            method='layer',
            center=True,
            scale=True
        )
        # Dropout between activations
        self.dropout = nn.Dropout(dropout)

        # Final convolution with zero initialization
        self.conv_out = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self,
                x: torch.Tensor,
                skip_h: torch.Tensor = None) -> torch.Tensor:
        # x, skip_h: (B, C, H, W)
        h = self.norm_in(x)
        if self.has_skip and skip_h is not None:
            skip_normed = self.norm_skip(skip_h)
            h = (h + skip_normed) / math.sqrt(2.0)

        # First activation + conv
        h = self.activation(h)
        h = self.conv_in(h)

        # Second normalization + activation + dropout
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.dropout(h)

        # Final conv and residual add
        h = self.conv_out(h)
        return x + h


if __name__ == "__main__":
    # Test without skip
    B, C, H, W = 2, 16, 32, 32
    x = torch.randn(B, C, H, W)
    block = SimpleResidualBlock(out_channels=C)
    y = block(x)
    print(f"No-skip mode: in={x.shape}, out={y.shape}")

    # Test with skip
    skip = torch.randn(B, C, H, W)
    block_skip = SimpleResidualBlock(out_channels=C, has_skip=True)
    y2 = block_skip(x, skip_h=skip)
    print(f"With-skip mode: in={x.shape}, out={y2.shape}")
