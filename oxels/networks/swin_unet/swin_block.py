import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock


class SwinBlock(nn.Module):
    """
    A wrapper for sequential Swin Transformer blocks (v1 or v2).

    Parameters
    ----------
    dim : int
        The number of input and output channels for the Swin Transformer blocks.
    input_res : Tuple[int, int]
        The input resolution of the feature map (height, width).
    shift_size : int, optional
        The shift size for the second block, by default 3.
    swin_version : str, optional
        Version of Swin Transformer to use: "v1" or "v2", by default "v2".

    Attributes
    ----------
    swtb1 : nn.Module
        First Swin Transformer block (non-shifted).
    swtb2 : nn.Module
        Second Swin Transformer block (shifted).
    """
    def __init__(self, dim: int, input_resolution: tuple[int, int], shift_size: int = 3, head_dim: int = 32, **kwargs):
        super().__init__()
        
        self.swtb1 = SwinTransformerBlock(
            dim=dim, 
            input_resolution=input_resolution, 
            head_dim=head_dim, 
            **kwargs
        )

        self.swtb2 = SwinTransformerBlock(
            dim=dim, 
            input_resolution=input_resolution, 
            head_dim=head_dim, 
            shift_size=shift_size, 
            **kwargs
        )

    def forward(self, x):
        """
        Forward pass through two sequential Swin Transformer blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W, C).

        Returns
        -------
        torch.Tensor
            Output tensor after two Swin Transformer blocks, of same shape as input.
        """
        return self.swtb2(self.swtb1(x))
