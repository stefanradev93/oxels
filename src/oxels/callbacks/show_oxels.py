import lightning as L
import torch
from torchvision.transforms.functional import resize
import wandb


class ShowOxels(L.Callback):
    def __init__(self, images: torch.Tensor, num_oxels=8, resolution=(64, 64), file_type="jpg", every_n_epochs=1, caption=None):
        super().__init__()
        self.images = torch.as_tensor(images)
        self.num_oxels = num_oxels
        self.resolution = resolution
        self.file_type = file_type
        self.every_n_epochs = every_n_epochs
        self.caption = caption if caption is not None else "Latent Space Visualization"

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        images = self.images.to(pl_module.device)
        # (batch_size, num_oxels, height, width)
        # also want to show oxels during finetuning and forward pass is different for combined model thus need to call
        # backbone directly
        oxels = pl_module.backbone(images)

        oxels = oxels[:, : self.num_oxels]
        oxels = torch.cat((images, oxels), dim=1)
        oxels = resize(oxels, self.resolution)

        # batch_size, channels, height, width = oxels.shape
        #
        # # normalize each image to [0, 1]
        # oxel_images = oxels.view((batch_size, channels, -1))
        # min_oxels = torch.min(oxel_images, dim=-1)[0][..., None, None]
        # max_oxels = torch.max(oxel_images, dim=-1)[0][..., None, None]
        # oxels = (oxels - min_oxels) / (max_oxels - min_oxels + 1e-8)
        #
        # # make a grid of shape (batch_size * height, channels * width)
        # oxels = oxels.permute(0, 2, 1, 3)
        # oxels = oxels.reshape((batch_size * height, channels * width))
        #
        # # ensure wandb knows these are grayscale
        # oxels = oxels[None]
        #
        # oxels = wandb.Image(oxels, caption=self.caption, file_type=self.file_type)
        #
        # wandb.log({self.caption: oxels, "trainer/global_step": trainer.global_step})

        B, C, H, W = oxels.shape
        rows = []
        for b in range(B):
            gray_imgs = []
            ch = oxels[b]  # (C, H, W)

            # --- 1) Make the RGB composite from the first 3 channels ---
            g = torch.stack([ch[0], ch[1], ch[2]], dim=0)  # (3, H, W)
            g = g.permute(1, 2, 0)  # (H, W, 3)
            gray_imgs.append(g)

            # --- 2) Turn each remaining single‐channel map into a 3‐channel gray image ---

            for i in range(3, C):
                g = ch[i]  # (H, W)
                g = torch.stack([g, g, g], dim=0)  # (3, H, W)
                g = g.permute(1, 2, 0)  # (H, W, 3)
                g = (g - g.min()) / (g.max() - g.min())  # Ensure values are in [0, 1]
                gray_imgs.append(g)

            # --- 3) Horizontally concatenate RGB + all the gray maps ---
            row = torch.concatenate(gray_imgs, dim=1)  # (H, W*(1 + C-3), 3)
            rows.append(row)

        # --- 4) If you want one big grid rather than a list, stack vertically: ---
        grid = torch.concatenate(rows, dim=0)  # (B*H, W*(C-2), 3)
        grid = grid.permute(2, 0, 1)  # (3, B*H, W*(C-2))
        oxels = wandb.Image(grid, caption=self.caption, file_type=self.file_type)

        wandb.log({self.caption: oxels, "trainer/global_step": trainer.global_step})
