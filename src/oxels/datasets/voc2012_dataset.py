import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from os import PathLike


class VOCDataset(Dataset):
    default_path = Path(__file__).parents[3] / "datasets"
    default_path = default_path.resolve()
    def __init__(self,
                 path: str | PathLike = default_path,
                 year="2012",
                 image_set="train",
                 download=False,
                 transform=None,
                 target_transform=None
                 ):
        self.path = Path(path) / "VOC"

        if transform is None:
            input_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        if target_transform is None:
            # Define target (segmentation mask) transformations
            # Using NEAREST interpolation to avoid smoothing label values
            target_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=Image.NEAREST),
                transforms.PILToTensor(),  # Produces 1 x H x W tensor with class indices
            ])

        self.voc_dataset = VOCSegmentation(
            root=self.path,
            year=year,
            image_set=image_set,
            download=download,
            transform=input_transform,
            target_transform=target_transform
        )

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        return image, target


if __name__ == '__main__':
    # Example  usage
    import matplotlib.pyplot as plt
    dataset = VOCDataset(year="2012", image_set="val", download=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for images, targets in dataloader:
        print(images.shape, targets.shape)  # Should print the shape of images and targets
        for i in range(len(images)):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i].permute(1, 2, 0))
        plt.show()
        # plot targets
        for i in range(len(targets)):
            plt.subplot(2, 4, i + 5)
            plt.imshow(targets[i].squeeze(), cmap='gray')
            plt.axis('off')
        plt.show()
        break  # Just to test one batch

