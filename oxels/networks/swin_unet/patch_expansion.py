import torch.nn as nn


class PatchExpansion(nn.Module):
    def __init__(self, dim: int, scale: int = 2, reduce_dim: bool = True):
        super().__init__()
        self.scale = scale
        self.out_dim = dim // scale if reduce_dim else dim

        self.expand = nn.Linear(dim, (scale**2) * self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)
        self.smooth = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1, groups=self.out_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, self.scale, self.scale, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5)  # (B, H, r, W, r, C)
        x = x.reshape(B, H * self.scale, W * self.scale, self.out_dim)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.smooth(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        return x
