import lightning as L
import torch
from torchvision.transforms.functional import resize
import wandb


class ShowOxels(L.Callback):
    def __init__(self, images: torch.Tensor, num_oxels=8, resolution=(64, 64), file_type="jpg"):
        super().__init__()
        self.images = torch.as_tensor(images)
        self.num_oxels = num_oxels
        self.resolution = resolution
        self.file_type = file_type

    def on_validation_epoch_end(self, trainer, pl_module):
        images = self.images.to(pl_module.device)
        # (batch_size, num_oxels, height, width)
        oxels = pl_module(images)

        oxels = oxels[:, : self.num_oxels]

        oxels = resize(oxels, self.resolution)

        batch_size, channels, height, width = oxels.shape

        # normalize each image to [0, 1]
        oxel_images = oxels.view((batch_size, channels, -1))
        min_oxels = torch.min(oxel_images, dim=-1)[0][..., None, None]
        max_oxels = torch.max(oxel_images, dim=-1)[0][..., None, None]
        oxels = (oxels - min_oxels) / (max_oxels - min_oxels + 1e-8)

        # make a grid of shape (batch_size * height, channels * width)
        oxels = oxels.permute(0, 2, 1, 3)
        oxels = oxels.reshape((batch_size * height, channels * width))

        # ensure wandb knows these are grayscale
        oxels = oxels[None]

        oxels = wandb.Image(oxels, caption="Latent Space Visualization", file_type=self.file_type)

        wandb.log({"oxels": oxels})
