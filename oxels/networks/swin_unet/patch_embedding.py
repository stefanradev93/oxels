import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings using a convolutional layer.

    Parameters
    ----------
    input_channels : int
        Number of channels in the input images.
    num_features : int
        Number of output channels (features) for each patch.
    patch_size : int
        Size of the square patches the image is divided into. Also used as both
        the kernel size and stride of the convolution.

    Attributes
    ----------
    conv : nn.Conv2d
        Convolutional layer that performs patch extraction and embedding.
    """

    def __init__(self, input_channels: int, embedding_dim: int, patch_size: int):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images):
        """
        Apply the patch embedding to a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_patches_h, num_patches_w, num_features),
            where each patch is represented by a feature vector.
        """
        return self.conv(images).permute(0, 2, 3, 1)
    