import lightning as L
import torch
import wandb


class ShowOxels(L.Callback):
    def __init__(self, images: torch.Tensor, num_oxels=8):
        super().__init__()
        self.images = torch.as_tensor(images)
        self.num_oxels = num_oxels

    def on_validation_epoch_end(self, trainer, pl_module):
        images = self.images.to(pl_module.device)
        # (batch_size, num_oxels, height, width)
        oxels = pl_module(images)

        oxels = oxels[:, : self.num_oxels]

        batch_size, channels, height, width = oxels.shape

        # make a grid of shape (batch_size * height, channels * width)
        oxels = oxels.permute(0, 2, 1, 3)
        oxels = oxels.reshape((batch_size * height, channels * width))

        # ensure wandb knows these are grayscale
        oxels = oxels[None]

        oxels = wandb.Image(oxels, caption="Latent Space Visualization")

        wandb.log({"oxels": oxels})
