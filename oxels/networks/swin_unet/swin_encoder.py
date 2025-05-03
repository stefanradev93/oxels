import torch.nn as nn

from torchvision.models.swin_transformer import PatchMerging

from .swin_block import SwinBlock


class SwinEncoder(nn.Module):
    """
    Swin-UNet encoder supporting arbitrary depth.

    Parameters
    ----------
    patch_embed_dim : int
        Number of output channels from the initial PatchEmbedding layer.
    input_resolution : tuple[int, int]
        Spatial resolution after patch embedding, e.g., (H//P, W//P) if patch size is P.
    num_stages : int
        Number of encoder stages (SwinBlock + PatchMerging pairs).
    """

    def __init__(
        self,
        patch_embed_dim: int,
        num_stages: int = 3,
        num_heads: int = 4,
        window_size: list[int] | int = 3,
        shift_size: list[int] | int = 3,
        use_conv: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.num_stages = num_stages
        self.swin_blocks = nn.ModuleList()
        self.patch_merge_blocks = nn.ModuleList()

        dim = patch_embed_dim

        for _ in range(num_stages):
            swin_block = SwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                use_conv=use_conv,
                **kwargs,
            )

            self.swin_blocks.append(swin_block)

            self.patch_merge_blocks.append(PatchMerging(dim))

            dim *= 2

    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W, C) — output from PatchEmbedding.

        Returns
        -------
        x : torch.Tensor
            Final downsampled representation.
        skip_features : List[torch.Tensor]
            List of features at each resolution for use in skip connections.
        """

        skip_features = []

        for swin_block, patch_merger in zip(self.swin_blocks, self.patch_merge_blocks):
            x = swin_block(x)
            skip_features.append(x)
            x = patch_merger(x)

        return x, skip_features
