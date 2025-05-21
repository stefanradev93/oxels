import torch
import torch.nn as nn


class SimpleUpSample(nn.Module):
    """
    Upsample by a factor of 2 via 1x1 conv + interpolation.
    1x1 conv used to match channels with possible skip connections and avoid projection conv somewhere else.
    """

    def __init__(self, in_channels: int, out_channels: int, interpolation: str = "nearest"):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True),
            nn.Upsample(scale_factor=2, mode=interpolation),
        )
        # Initialize the 1x1 conv
        nn.init.xavier_uniform_(self.op[0].weight, gain=1.0)
        nn.init.zeros_(self.op[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


if __name__ == "__main__":
    # Test
    x = torch.randn(2, 16, 32, 32)
    up = SimpleUpSample(in_channels=16, out_channels=8)
    y = up(x)
    print(f"Input shape: {x.shape} -> UpSample output: {y.shape}")
