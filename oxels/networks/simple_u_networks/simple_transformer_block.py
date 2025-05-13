import torch
import torch.nn as nn

from oxels.typing import ActivationType
from simple_mlp  import SimpleMLPBlock
from simple_self_attention import SimpleAttention

class SimpleTransformerBlock(nn.Module):
    """
    Combines the SimpleMLPBlock and SimpleAttention in a residual fashion:
      x = x + MLP(x)
      x = x + Attention(x)

    Both sub-blocks do not have time or text conditioning in this version.
    """
    def __init__(self,
                 channel_dim: int,
                 expansion: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 activation: ActivationType=nn.SiLU,
                 ):
        super(SimpleTransformerBlock, self).__init__()
        self.mlp  = SimpleMLPBlock(
            channel_dim=channel_dim,
            expansion=expansion,
            dropout=dropout,
            activation=activation
        )
        self.attn = SimpleAttention(
            num_heads=num_heads,
            channel_dim=channel_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # Residual MLP
        x = x + self.mlp(x)
        # Residual Self-Attention
        x = x + self.attn(x)
        return x

if __name__ == "__main__":
    # Simple test
    B, C, H, W = 2, 16, 32, 32
    x = torch.randn(B, C, H, W)
    block = SimpleTransformerBlock(channel_dim=C)
    y = block(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")