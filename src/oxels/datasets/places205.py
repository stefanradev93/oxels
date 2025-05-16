from pathlib import Path
from os import PathLike
from typing import Literal

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, extract_archive, check_integrity

import deeplake


class Places205Dataset(Dataset):
    url = "https://data.csail.mit.edu/places/places205/"

    file_hashes = {
        "train_256.tar.gz": "<placeholder>",
        "val_256.tar.gz": "<placeholder>",
    }

    default_path = Path(__file__).parents[2] / "datasets"
    default_path = default_path.resolve()

    def __init__(
        self,
        path: str | PathLike = default_path,
        split: Literal["training", "validation"] = "training",
    ):
        self.path = Path(path) / "Places205"

        self.download(self.path)

        os.makedirs(self.root, exist_ok=True)
        # Perform download, integrity check, and extraction
        self.download()

        # Labels file
        labels_file = os.path.join(self.root, f"{self.split}_labels.txt")
        if not os.path.isfile(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}.")

        # Build sample list
        self.samples = []
        with open(labels_file, "r") as f:
            for line in f:
                rel_path, label = line.strip().split()
                img_path = os.path.join(self.root, "images", self.split, rel_path)
                self.samples.append((img_path, int(label)))

    def download(self, path: Path):
        deeplake.open("hub://activeloop/places205")
        for file, target_hash in self.file_hashes.items():
            url = self.url + file
            filepath = path / file

            if not filepath.is_file():
                download_url(url, path, file, target_hash)

            if not check_integrity(filepath, target_hash):
                raise RuntimeError(f"File {filepath} not found or corrupted.")
            else:
                print(f"Found and verified {file} at {path} with hash {target_hash}.")

            extract_archive(filepath, path)

        # Extract the archive
        for file in self.file_hashes.keys():
            archive_path = path / file
            if archive_path.is_file():
                extract_archive(archive_path, path)
                os.remove(archive_path)
        archive_name = os.path.basename(self.url)
        archive_path = os.path.join(self.root, archive_name)

        # Download if missing or fails integrity
        if not check_integrity(archive_path, md5=self.md5):
            download_url(url=self.url, download_root=self.root, filename=archive_name, md5=self.md5)
        # Extract regardless (extract_archive will skip already extracted files)
        extract_archive(from_path=archive_path, to_path=self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# Example usage:
# from torchvision import transforms
#
# transform_train = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])
#
# dataset = Places205Dataset(
#     root='/path/to/places205',
#     split='train',
#     transform=transform_train,
#     download_url='http://data.csail.mit.edu/places/places205/train_256.tar.gz',
#     md5='f880c7af5c36e0a01b758e5198d6e8f9'  # replace with actual checksum
# )
# loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
