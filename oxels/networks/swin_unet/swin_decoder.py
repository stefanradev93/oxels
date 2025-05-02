import torch
import torch.nn as nn

from .swin_block import SwinBlock
from .patch_expansion import PatchExpansion


class SwinDecoder(nn.Module):
    """
    Swin-UNet decoder supporting arbitrary depth.

    Parameters
    ----------
    patch_embed_dim : int
        Base feature dimension used in the encoder (same as patch_embed_dim).
    input_resolution : tuple[int, int]
        Spatial resolution after patch embedding (H//P, W//P).
    num_stages : int
        Number of decoder stages (must match encoder depth).
    """

    def __init__(
        self, 
        patch_embed_dim: int,
        num_stages: int = 3, 
        num_heads: int = 4, 
        window_size: int = 3, 
        shift_size: int = 3,
        use_conv: bool = True, 
        **kwargs
    ):
        super().__init__()

        dim = patch_embed_dim * (2 ** num_stages)

        self.patch_expand_blocks = nn.ModuleList()
        self.swin_blocks = nn.ModuleList()
        self.concat_linears = nn.ModuleList()

        for _ in range(num_stages):

            self.patch_expand_blocks.append(PatchExpansion(dim))

            self.concat_linears.append(nn.Linear(dim, dim//2))

            swin_block = SwinBlock(
                dim=dim // 2, 
                num_heads=num_heads, 
                window_size=window_size, 
                shift_size=shift_size, 
                use_conv=use_conv, 
                **kwargs
            )

            self.swin_blocks.append(swin_block)

            dim = dim // 2

    def forward(self, x, skip_features):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x : torch.Tensor
            Final latent representation from the encoder, shape (B, H, W, C).
        skip_features : list of torch.Tensor
            List of encoder skip features in **same order** they were stored during encoding.

        Returns
        -------
        torch.Tensor
            Final upsampled representation.
        """
        # Use skip features in reverse order
        for patch_expand, concat_proj, swin_block, skip in zip(
            self.patch_expand_blocks,
            self.concat_linears,
            self.swin_blocks,
            reversed(skip_features)
        ):
            x = patch_expand(x)
            x = torch.cat([x, skip], dim=-1)
            x = concat_proj(x)
            x = swin_block(x)
        return x
