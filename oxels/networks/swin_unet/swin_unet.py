import torch.nn as nn

from .patch_embedding import PatchEmbedding
from .patch_expansion import PatchExpansion
from .swin_block import SwinBlock
from .swin_encoder import SwinEncoder
from .swin_decoder import SwinDecoder


class SwinUNet(nn.Module):
    """
    Swin U-Net architecture for dense prediction tasks (e.g., segmentation).

    This model uses a hierarchical Swin Transformer-based encoder-decoder structure,
    with patch embedding and patch merging/expansion to process inputs at multiple resolutions.

    Parameters
    ----------
    height : int
        Height of the input image (must be divisible by patch_size * 2^num_stages).
    width : int
        Width of the input image (must be divisible by patch_size * 2^num_stages).
    channels : int
        Number of channels in the input image (e.g., 3 for RGB).
    patch_embed_dim : int, optional
        Number of output channels from the initial PatchEmbedding layer. Default is 32.
    patch_size : int, optional
        Patch size for the initial embedding. Default is 4.
    num_stages : int, optional
        Number of encoder/decoder stages. Default is 3.
    head_dim : int, optional
        Head dimension used in Swin blocks. Default is 32.
    shift_size : int, optional
        Shift size used in shifted window attention (if applicable). Default is 3.
    oxel_dim : int, optional
        Number of output channels from the final convolutional head. Default is 16.
    **kwargs : dict
        Additional keyword arguments passed to Swin blocks (e.g., mlp_ratio, dropout).
    """

    ...
    def __init__(
        self, 
        height: int, 
        width: int, 
        channels: int, 
        patch_embed_dim: int = 32, 
        patch_size: int = 4, 
        num_stages: int = 3, 
        head_dim: int = 32,
        shift_size: int = 3,
        oxel_dim: int = 16,
        **kwargs
    ):

        super().__init__()

        self.patch_embed = PatchEmbedding(channels, patch_embed_dim, patch_size)

        self.encoder = SwinEncoder(
            patch_embed_dim, 
            input_resolution=(height//patch_size, width//patch_size), 
            shift_size=shift_size,
            num_stages=num_stages,
            **kwargs
        )

        self.bottleneck = SwinBlock(
            dim=patch_embed_dim*(2**num_stages), 
            input_resolution=(
                height // (patch_size * 2**num_stages), 
                width  // (patch_size * 2**num_stages)
            ),
            shift_size=shift_size, 
            head_dim=head_dim,
            **kwargs
        )

        self.decoder = SwinDecoder(
            patch_embed_dim, 
            input_resolution=(height//patch_size, width//patch_size), 
            shift_size=shift_size,
            num_stages=num_stages,
            **kwargs
        )
        
        self.final_patch_expansion = PatchExpansion(dim=patch_embed_dim, scale=patch_size, reduce_dim=False)

        self.head = nn.Conv2d(patch_embed_dim, oxel_dim, kernel_size=1, padding='same')

    def forward(self, x):
        """
        Forward pass of the SwinUNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W), where:
            - B is the batch size,
            - C is the number of input channels (e.g., 3 for RGB),
            - H and W are the spatial dimensions (height and width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, oxel_dim, H, W), where `oxel_dim` is the number
            of output channels defined in the final convolutional layer (e.g., number of segmentation classes).
        """
        x = self.patch_embed(x)
        
        x, skip_features  = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x, skip_features)
        
        x = self.final_patch_expansion(x)
        
        x = self.head(x.permute(0,3,1,2))

        return x