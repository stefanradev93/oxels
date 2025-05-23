from os import PathLike
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as ImageNetVision
from torchvision.datasets.utils import check_integrity, download_url, extract_archive

from oxels.transforms import PerspectiveTransform


class ImageNet(Dataset):
    url = "https://image-net.org/data/ILSVRC/2012/"

    file_hashes = {
        "ILSVRC2012_devkit_t12.tar.gz": "fa75699e90414af021442c21a62c3abf",
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
        transform=None,
    ):
        super().__init__()
        self.path = Path(path) / "ImageNet1K"
        self.split = split
        self.download(self.path)
        self.dataset = ImageNetVision(self.path, split, transform=transform)
        self.transform = PerspectiveTransform(w=w, h=h, frac_keep=frac_keep)

    def __getitem__(self, item):
        rgb, label = self.dataset[item]
        view1, view2, indices, flags, mask1, mask2 = self.transform.get_views_and_permutation(rgb)
        return view1, view2, label, indices, flags, mask1, mask2

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
