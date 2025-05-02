import torch.nn as nn

from torchvision.models.swin_transformer import SwinTransformerBlock


class SwinBlock(nn.Module):
    """
    A wrapper for sequential Swin Transformer blocks (v1 or v2).

    Parameters
    ----------
    dim : int
        The number of input and output channels for the Swin Transformer blocks.
    shift_size : int, optional
        The shift size for the second block, by default 3.
    head_dim: int, optional
        The dimension of the heads used
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 4, 
        window_size: int = 3, 
        shift_size: int = 3, 
        use_conv: bool = True, 
        **kwargs
    ):
        super().__init__()
        
        self.swtb1 = SwinTransformerBlock(
            dim=dim, 
            num_heads=num_heads,
            window_size=[window_size, window_size],
            shift_size=[0, 0],
            **kwargs
        )

        self.swtb2 = SwinTransformerBlock(
            dim=dim, 
            num_heads=num_heads,
            window_size=[window_size, window_size],
            shift_size=[shift_size, shift_size],
            **kwargs
        )
        
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # Depthwise
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise
            )

        self.use_conv = use_conv

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

        x_swin = self.swtb1(x)
        x_swin = self.swtb2(x_swin)

        if self.use_conv:
            # Convert to (B, C, H, W)
            x_conv = x.permute(0, 3, 1, 2).contiguous()
            x_conv = self.conv(x_conv)
            x_conv = x_conv.permute(0, 2, 3, 1).contiguous()

            # Fuse Swin + Conv
            x_out = x_swin + x_conv
        else:
            x_out = x_swin

        return x_out
