from os import PathLike
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.imagenet import parse_train_archive, parse_val_archive
from torchvision.datasets.utils import check_integrity, download_url, extract_archive
from torchvision.transforms import ToTensor

from oxels.transforms import PerspectiveTransform


class ImageNet(Dataset):
    url = "https://image-net.org/data/ILSVRC/2012/"

    file_hashes = {
        "ILSVRC2012_img_train.tar": "1d675b47d978889d74fa0da5fadfb00e",
        "ILSVRC2012_img_val.tar": "29b22e2961454d5413ddabcf34fc5622",
    }

    default_path = Path(__file__).parents[3] / "datasets"
    default_path = default_path.resolve()

    def __init__(
        self,
        path: str | PathLike = default_path,
        w: int = 256,
        h: int = 256,
        frac_keep: float = 0.125,
        split: str = "train",
        transform=ToTensor(),
    ):
        super().__init__()
        self.path = Path(path) / "ImageNet1K"
        self.split = split
        self.download(self.path)
        self.dataset = ImageFolder(self.path / self.split, transform=transform)
        self.transform = PerspectiveTransform(w=w, h=h, frac_keep=frac_keep)

    def __getitem__(self, item):
        with torch.device("cpu"):
            rgb, label = self.dataset[item]
        rgb = rgb.numpy()
        rgb = rgb.transpose(1, 2, 0)

        # this needs channels last
        view1, view2, permutation, flags, mask1, mask2 = self.transform.get_views_and_permutation(rgb)

        view1 = view1.transpose(2, 0, 1)
        view2 = view2.transpose(2, 0, 1)
        return view1, view2, permutation, flags, mask1, mask2

    def __len__(self):
        return len(self.dataset)

    def download(self, path):
        path.mkdir(parents=True, exist_ok=True)

        if (path / self.split).is_dir():
            print(f"ImageNet1K split {self.split} already exists at {path}. Skipping download.")
            return

        for file, target_hash in self.file_hashes.items():
            url = self.url + file
            filepath = path / file

            if not filepath.is_file():
                download_url(url, path, file, target_hash)

            if not check_integrity(filepath, target_hash):
                raise RuntimeError(f"File {filepath} not found or corrupted.")
            else:
                print(f"Found and verified {file} at {path} with hash {target_hash}.")

        # let torchvision extract the dataset
        parse_train_archive(path, "ILSVRC2012_img_train.tar")
        parse_val_archive(path, "ILSVRC2012_img_val.tar")
