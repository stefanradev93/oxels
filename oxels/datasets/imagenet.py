from os import PathLike
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as ImageNetVision
from torchvision.datasets.utils import calculate_md5, check_integrity, download_url

from oxels.transforms import PerspectiveTransform
from oxels.utils import md5
import requests
import gzip
import tarfile
import shutil

import fiftyone.zoo


class ImageNet(Dataset):
    url = "https://image-net.org/data/ILSVRC/2012/"

    file_hashes = {
        "ILSVRC2012_devkit_t12.tar.gz": "fa75699e90414af021442c21a62c3abf",
        "ILSVRC2012_img_train.tar": "1d675b47d978889d74fa0da5fadfb00e",
        "ILSVRC2012_img_val.tar": "29b22e2961454d5413ddabcf34fc5622",
    }

    default_path = Path(__file__).parents[2] / "datasets"
    default_path = default_path.resolve()

    def __init__(self, path: str | PathLike = default_path, w: int = 256, h: int = 256, frac_keep: float = 0.125, split: str = "train"):
        super().__init__()
        self.path = Path(path) / "ImageNet1K"
        self.download(self.path)
        self.dataset = ImageNetVision(self.path, split)
        self.transform = PerspectiveTransform(w=w, h=h, frac_keep=frac_keep)

    def __getitem__(self, item):
        rgb, label = self.dataset[item]
        view1, view2, indices, flags, mask1, mask2 = self.transform.get_views_and_permutation(rgb)
        return view1, view2, label, indices, flags, mask1, mask2

    def download(self, path):
        path.mkdir(parents=True, exist_ok=True)

        for file, target_hash in self.file_hashes.items():
            url = self.url + file
            filepath = path / file

            if not filepath.is_file():
                download_url(url, path, file, target_hash)

            if not check_integrity(filepath, target_hash):
                raise RuntimeError(f"File {filepath} not found or corrupted.")
            else:
                print(f"Found and verified {file} at {path} with hash {target_hash}.")

        fiftyone.zoo.download_zoo_dataset("imagenet-2012", source_dir=path)
