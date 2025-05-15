import torch
import torch.nn as nn


class SimpleDownSample(nn.Module):
    """
    Downsample by a factor of 2 via AvgPool + 1x1 conv.
    1x1 conv used to match channels with possible skip connections and avoid projection conv somewhere else.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True),
        )
        # Initialize the 1x1 conv
        nn.init.xavier_uniform_(self.op[1].weight, gain=1.0)
        nn.init.zeros_(self.op[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


if __name__ == "__main__":
    # Test
    x = torch.randn(2, 16, 64, 64)
    down = SimpleDownSample(in_channels=16, out_channels=32)
    y = down(x)
    print(f"Input shape: {x.shape} -> DownSample output: {y.shape}")
