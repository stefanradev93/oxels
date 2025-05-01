import torch.nn as nn


class PatchExpansion(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim//2)
        self.expand = nn.Linear(dim, 2*dim, bias=False)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, 2, 2, C//4)
        x = x.permute(0, 1, 3, 2, 4, 5)

        x = x.reshape(B, H*2, W*2, C//4)

        x = self.norm(x)
        return x
