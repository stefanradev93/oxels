import torch.nn as nn


class PatchExpansion(nn.Module):
    """
    Generalized patch expansion layer to upsample spatial resolution by a factor
    of 2× or 4×, with optional reduction of feature dimension.

    Parameters
    ----------
    dim : int
        Number of input channels per patch (typically from PatchMerging).
    scale : int, optional
        Upsampling factor for spatial dimensions. Supported values: 2 or 4. Default is 2.
    out_dim : int, optional
        Output feature dimension after expansion. If None and reduce_dim=True, it defaults to dim // scale.
        If None and reduce_dim=False, it defaults to dim.
    reduce_dim : bool, optional
        If True, reduces the channel dimension by a factor of `scale`.
        If False, preserves the original feature dimension. Default is False.

    Attributes
    ----------
    expand : nn.Linear
        Linear layer to expand feature dimensions before reshaping.
    norm : nn.LayerNorm
        Layer normalization applied after reshaping and upsampling.
    """

    def __init__(self, dim: int, scale: int = 2, reduce_dim: bool = True):
        super().__init__()

        self.scale = scale

        if reduce_dim:
            self.out_dim = dim // scale
        else:
            self.out_dim = dim

        self.expand = nn.Linear(dim, (scale ** 2) * self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        """
        Forward pass for patch expansion.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W, C), where:
            - B is batch size,
            - H and W are spatial dimensions,
            - C is the input feature dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, scale*H, scale*W, out_dim), where
            - spatial size is upsampled by `scale`,
            - channel dimension is `out_dim`.
        """
        B, H, W, C = x.shape
        x = self.expand(x)

        x = x.view(B, H, W, self.scale, self.scale, self.out_dim) 
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H * self.scale, W * self.scale, self.out_dim)

        x = self.norm(x)
        return x
