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
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, every_n_epochs=1, caption=None, colormap=None):
        super().__init__()
        self.images = torch.as_tensor(images)
        self.labels = torch.as_tensor(labels)
        self.every_n_epochs = every_n_epochs
        self.caption = caption if caption is not None else "Segmentation Visualization"
        self.colormap = VOC_COLORMAP if colormap is None else colormap

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


ADE20K_CLASS_LABELS = {
    0: 'wall', 1: 'building', 2: 'sky', 3: 'floor', 4: 'tree', 5: 'ceiling', 6: 'road', 7: 'bed',
    8: 'windowpane', 9: 'grass', 10: 'cabinet', 11: 'sidewalk', 12: 'person', 13: 'earth', 14: 'door',
    15: 'table', 16: 'mountain', 17: 'plant', 18: 'curtain', 19: 'chair', 20: 'car', 21: 'water',
    22: 'painting', 23: 'sofa', 24: 'shelf', 25: 'house', 26: 'sea', 27: 'mirror', 28: 'rug',
    29: 'field', 30: 'armchair', 31: 'seat', 32: 'fence', 33: 'desk', 34: 'rock', 35: 'wardrobe',
    36: 'lamp', 37: 'bathtub', 38: 'railing', 39: 'cushion', 40: 'base', 41: 'box', 42: 'column',
    43: 'signboard', 44: 'chest of drawers', 45: 'counter', 46: 'sand', 47: 'sink', 48: 'skyscraper',
    49: 'fireplace', 50: 'refrigerator', 51: 'grandstand', 52: 'path', 53: 'stairs', 54: 'runway',
    55: 'case', 56: 'pool table', 57: 'pillow', 58: 'screen door', 59: 'stairway', 60: 'river',
    61: 'bridge', 62: 'bookcase', 63: 'blind', 64: 'coffee table', 65: 'toilet', 66: 'flower',
    67: 'book', 68: 'hill', 69: 'bench', 70: 'countertop', 71: 'stove', 72: 'palm', 73: 'kitchen island',
    74: 'computer', 75: 'swivel chair', 76: 'boat', 77: 'bar', 78: 'arcade machine', 79: 'hovel',
    80: 'bus', 81: 'towel', 82: 'light', 83: 'truck', 84: 'tower', 85: 'chandelier', 86: 'awning',
    87: 'streetlight', 88: 'booth', 89: 'television receiver', 90: 'airplane', 91: 'dirt track',
    92: 'apparel', 93: 'pole', 94: 'land', 95: 'bannister', 96: 'escalator', 97: 'ottoman',
    98: 'bottle', 99: 'buffet', 100: 'poster', 101: 'stage', 102: 'van', 103: 'ship', 104: 'fountain',
    105: 'conveyer belt', 106: 'canopy', 107: 'washer', 108: 'plaything', 109: 'swimming pool',
    110: 'stool', 111: 'barrel', 112: 'basket', 113: 'waterfall', 114: 'tent', 115: 'bag',
    116: 'minibike', 117: 'cradle', 118: 'oven', 119: 'ball', 120: 'food', 121: 'step', 122: 'tank',
    123: 'trade name', 124: 'microwave', 125: 'pot', 126: 'animal', 127: 'bicycle', 128: 'lake',
    129: 'dishwasher', 130: 'screen', 131: 'blanket', 132: 'sculpture', 133: 'hood', 134: 'sconce',
    135: 'vase', 136: 'traffic light', 137: 'tray', 138: 'ashcan', 139: 'fan', 140: 'pier',
    141: 'crt screen', 142: 'plate', 143: 'monitor', 144: 'bulletin board', 145: 'shower',
    146: 'radiator', 147: 'glass', 148: 'clock', 149: 'flag'
}


class ShowSegmentationWandb(L.Callback):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, every_n_epochs=1, caption=None, class_dictionary=None):
        super().__init__()
        self.images = torch.as_tensor(images)
        self.labels = torch.as_tensor(labels)
        self.every_n_epochs = every_n_epochs
        self.caption = caption if caption is not None else "Segmentation Visualization"
        self.class_dict = ADE20K_CLASS_LABELS if class_dictionary is None else class_dictionary

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        images = self.images.to(pl_module.device)
        labels = self.labels.to(pl_module.device).long().squeeze(1)

        logits = pl_module(images)
        preds = torch.argmax(logits, dim=1)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denorm_images = denormalize(images, mean, std).clamp(0, 1)

        wandb_images = []
        for i in range(images.shape[0]):
            img = denorm_images[i].permute(1, 2, 0).cpu().numpy()
            gt_mask = labels[i].cpu().numpy()
            pred_mask = preds[i]
            if pred_mask.shape != labels[i].shape:
                pred_mask = resize(pred_mask.unsqueeze(0).float(), labels[i].shape, antialias=False).long().squeeze(0)
            pred_mask = pred_mask.cpu().numpy()

            wandb_images.append(wandb.Image(
                img,
                masks={
                    "ground_truth": {
                        "mask_data": gt_mask,
                        "class_labels": self.class_dict,
                    },
                    "predictions": {
                        "mask_data": pred_mask,
                        "class_labels": self.class_dict,
                    }
                },
                caption=f"{self.caption} #{i}"
            ))
        wandb.log({self.caption: wandb_images, "trainer/global_step": trainer.global_step})