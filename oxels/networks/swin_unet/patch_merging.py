import torch.nn as nn


class PatchMerging(nn.Module):
    """
    Patch merging layer used in hierarchical vision transformers (e.g., Swin Transformer).

    This layer downsamples the spatial resolution by a factor of 2 while increasing
    the feature dimension. It merges 2x2 neighboring patches and applies a linear projection.

    Parameters
    ----------
    dim : int
        Number of input channels per patch before merging.

    Attributes
    ----------
    norm : nn.LayerNorm
        Layer normalization applied after patch merging.
    reduction : nn.Linear
        Linear layer that reduces the concatenated features from 4*dim to 2*dim.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        Forward pass for patch merging.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W, C), where:
            - B is the batch size,
            - H and W are spatial dimensions (height and width),
            - C is the number of input channels.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, H//2, W//2, 2*C), with downsampled spatial dimensions
            and doubled channel count.
        """
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x
