
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.transforms import ToTensor
from pathlib import Path
from os import PathLike

from oxels.transforms import PerspectiveTransform


class CIFAR10(Dataset):
    default_path = Path(__file__).parents[2] / "datasets"
    default_path = default_path.resolve()

    def __init__(self, path: str | PathLike = default_path, w=32, h=32, frac_keep=0.125, split="train"):
        super().__init__()
        self.path = Path(path) / "CIFAR-10"
        self.dataset = _CIFAR10(self.path, train=(split == "train"), transform=ToTensor(), download=True)
        self.transform = PerspectiveTransform(w=w, h=h, frac_keep=frac_keep)

    def __getitem__(self, item):
        with torch.device("cpu"):
            rgb, _ = self.dataset.__getitem__(item)
        rgb = rgb.numpy()
        rgb = rgb.transpose(1, 2, 0)

        # this needs channels last
        view1, view2, permutation, flags, mask1, mask2 = self.transform.get_views_and_permutation(rgb)

        view1 = view1.transpose(2, 0, 1)
        view2 = view2.transpose(2, 0, 1)
        return view1, view2, permutation, flags, mask1, mask2

    def __len__(self):
        return len(self.dataset)
