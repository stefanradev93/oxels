import torch
import lightning as L
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


class ComputeOxelSegmentationStat(L.Callback):
    def __init__(self, dataloader, n_images, n_most_frequent, every_n_epochs=1, caption=None, class_dictionary=None):
        super().__init__()
        self.dataloader = dataloader
        self.n_images = n_images
        self.n_most_frequent = n_most_frequent
        self.every_n_epochs = every_n_epochs
        self.caption = caption if caption is not None else "Segmentation Loss Proxy"
        self.class_dict = ADE20K_CLASS_LABELS if class_dictionary is None else class_dictionary

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        print(f"Computing segmentation proxy for {self.n_images} images...")
        all_boxels, all_labels = [], []
        image_counter = 0
        for images, labels in self.dataloader:
            image_counter += images.shape[0]
            images = images.to(pl_module.device)
            labels = labels.to(pl_module.device).long().squeeze(1)

            oxels = pl_module.backbone(images) # (batch_size, num_oxels, height, width)
            oxels = oxels.cpu().numpy()
            num_oxels = oxels.shape[1]
            boxels = sum([2 ** i * np.array(oxels[:, i] > 0, dtype=np.uint32) for i in range(num_oxels)])
            all_boxels.append(boxels)
            all_labels.append(labels.cpu().numpy())
            if image_counter > self.n_images:
                break
        all_boxels = np.concatenate(all_boxels, axis=0)[: self.n_images]
        all_labels = np.concatenate(all_labels, axis=0)[: self.n_images]

        tp_ratios = {}
        for SAMPLE_LABEL, _ in ADE20K_CLASS_LABELS.items():
            sample_boxels = all_boxels[np.where(all_labels == SAMPLE_LABEL)]
            unique_sample_boxels, sample_counts = np.unique(sample_boxels, return_counts=True)
            unique_sample_boxels = unique_sample_boxels[np.argsort(sample_counts)]
            sample_counts = np.sort(sample_counts)
            true_positives = 0
            all_positives = 0
            num_most_frequent = min(self.n_most_frequent, len(unique_sample_boxels))
            for i in range(-1, -num_most_frequent - 1, -1):  # go over 10 most frequent oxels associated with the sample label
                true_positives += sample_counts[i]
                b = unique_sample_boxels[i]
                all_positives += np.sum(all_boxels == b)
            tp_ratios[SAMPLE_LABEL] = true_positives / all_positives if all_positives > 0 else 0
            # print(f"{ADE20K_CLASS_LABELS[SAMPLE_LABEL]}: {true_positives/all_positives:.3f}")
        fig = plt.figure(figsize=(20, 5))
        plt.bar(range(len(tp_ratios)), list(tp_ratios.values()), align='center')
        plt.xticks(range(len(tp_ratios)), list(ADE20K_CLASS_LABELS.values()), rotation=90)
        plt.xlabel("ADE20K Class Labels")
        plt.ylabel("True Positive Ratio")
        plt.title("True Positive Ratios for ADE20K Class Labels")
        plt.tight_layout()

        wandb.log({
            self.caption: wandb.Image(fig),
            "trainer/global_step": trainer.global_step
        })
        plt.close(fig)
        del all_boxels, all_labels, fig
