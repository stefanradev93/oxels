import torch
import lightning as L
import wandb
from torchvision.transforms.functional import resize, to_pil_image
import numpy as np

# PASCAL VOC2012 Color Map
VOC_COLORMAP = torch.tensor([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
], dtype=torch.uint8)

def denormalize(tensor, mean, std):
    """Denormalize a tensor image with mean and standard deviation."""
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.clone() * std + mean

def mask_to_rgb(mask, colormap):
    """Converts a segmentation mask to an RGB image."""
    # The VOC dataset uses 255 as an ignore-index. Map it to the background color.
    mask_for_viz = mask.cpu().clone().long()
    mask_for_viz[mask_for_viz == 255] = 0
    rgb_mask = colormap[mask_for_viz]
    return rgb_mask.permute(2, 0, 1)

class ShowSegmentation(L.Callback):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, every_n_epochs=1, caption=None):
        super().__init__()
        self.images = torch.as_tensor(images)
        self.labels = torch.as_tensor(labels)
        self.every_n_epochs = every_n_epochs
        self.caption = caption if caption is not None else "Segmentation Visualization"
        self.colormap = VOC_COLORMAP

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        
        images = self.images.to(pl_module.device)
        labels = self.labels.to(pl_module.device)

        logits = pl_module(images)
        preds = torch.argmax(logits, dim=1)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denorm_images = denormalize(images, mean, std)
        denorm_images = (denorm_images.clamp(0, 1) * 255).to(torch.uint8)

        labels = labels.squeeze(1)

        grid_images = []
        for i in range(images.shape[0]):
            img = denorm_images[i]
            
            # Use original label which has correct spatial dimensions
            gt_mask_rgb = mask_to_rgb(labels[i], self.colormap)
            
            # Resize prediction to match label size for visualization
            pred_resized = resize(preds[i].unsqueeze(0).float(), list(labels.shape[1:]), antialias=False).long().squeeze(0)
            pred_mask_rgb = mask_to_rgb(pred_resized, self.colormap)
            
            h, w = img.shape[1], img.shape[2]
            gt_mask_rgb = resize(gt_mask_rgb, [h, w], antialias=False)
            pred_mask_rgb = resize(pred_mask_rgb, [h, w], antialias=False)

            combined = torch.cat([img.cpu(), gt_mask_rgb, pred_mask_rgb], dim=2)
            grid_images.append(combined)

        grid = torch.cat(grid_images, dim=1)
        
        wandb.log({
            self.caption: wandb.Image(grid),
            "trainer/global_step": trainer.global_step
        }) 