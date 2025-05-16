import torch
import torch.nn as nn


class SimpleNorm(nn.Module):
    """
    PyTorch SimpleNorm.

    This module supports two normalization strategies on channel dimension:
      - method='group': GroupNorm with a specified number of groups.
      - method='layer': LayerNorm-like behavior via GroupNorm with a single group (equivalent to Keras LayerNormalization over channels).

    Learnable parameters:
      - center (bool): if True, adds a learnable bias term ("NormalizeWithBias").
      - scale  (bool): if True, adds a learnable weight term (scale).

    Semantics:
      * Normalize (no bias):     method=<...>, center=False, scale=True
      * NormalizeWithBias:        method=<...>, center=True,  scale=True
      * Disable all parameters:   center=False, scale=False

    Arguments:
        channel_dim (int): Number of channels C.
        method (str):      'group' or 'layer'.
        groups (int):      Number of groups if using GroupNorm (method='group').
        center (bool):     Learnable bias (True=add offset).
        scale (bool):      Learnable scale (True=add weight).
        eps (float):       Small epsilon for numerical stability.
    """

    def __init__(
        self,
        channel_dim: int,
        method: str = "group",
        groups: int = 8,
        center: bool = True,
        scale: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        if method == "group":
            ng = groups
        elif method == "layer":
            ng = 1
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # affine covers both weight (scale) and bias (center)
        self.norm = nn.GroupNorm(num_groups=ng, num_channels=channel_dim, eps=eps, affine=(center or scale))

        # disable weight / bias gradients if requested
        if not scale:
            with torch.no_grad():
                self.norm.weight.fill_(1.0)
            self.norm.weight.requires_grad = False
        if not center:
            with torch.no_grad():
                self.norm.bias.fill_(0.0)
            self.norm.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, ...) where ... are spatial dimensions
        return self.norm(x)


if __name__ == "__main__":
    # Simple executable test
    batch, channels, height, width = 2, 16, 32, 32
    x = torch.randn(batch, channels, height, width)
    for method in ["group", "layer"]:
        norm = SimpleNorm(channel_dim=channels, method=method, groups=4)
        y = norm(x)
        print(f"method={method}, output shape: {y.shape}")
