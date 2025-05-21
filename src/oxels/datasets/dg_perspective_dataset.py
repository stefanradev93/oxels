import torch
from torch.utils.data import Dataset, random_split
from typing import Literal
from pathlib import Path
from os import PathLike

from oxels.transforms import PerspectiveTransform

from .dg import MultiDomainDataset


class DGPerspectiveDataset(Dataset):
    splits = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
    }

    default_path = Path(__file__).parents[3] / "datasets"
    default_path = default_path.resolve()

    def __init__(
        self,
        *,
        path: str | PathLike = default_path,
        dataset: str = "camelyon17",
        w: int = 256,
        h: int = 256,
        frac_keep: float = 0.125,
        split: Literal["train", "val", "test"] = "train",
        domain_split: Literal["id", "ood"] = "id",
        seed: int = 0,
        augmentations: callable = None,
    ):
        self.path = Path(path)
        self.split = split
        self.domain_split = domain_split
        self.augmentations = augmentations
        self.dataset = self._get_dataset(dataset, seed=seed)
        self.transform = PerspectiveTransform(w=w, h=h, frac_keep=frac_keep)

    def __getitem__(self, item):
        with torch.device("cpu"):
            rgb, label, domain = self.dataset.__getitem__(item)
        rgb = rgb.numpy(force=True)
        rgb = rgb.transpose(1, 2, 0)

        # this needs channels last
        view1, view2, permutation, flags, mask1, mask2 = self.transform.get_views_and_permutation(rgb)

        view1 = view1.transpose(2, 0, 1)
        view2 = view2.transpose(2, 0, 1)
        return view1, view2, permutation, flags, mask1, mask2

    def __len__(self):
        return len(self.dataset)

    def _get_raw_dataset(self, dataset: str):
        match dataset.lower():
            case "camelyon17":
                from .dg import Camelyon17

                return Camelyon17(self.path, download=True)
            case "coloredmnist":
                from .dg import ColoredMNIST

                return ColoredMNIST(self.path, download=True, transform=self.augmentations)
            case "fmowregion":
                from .dg import FMoWRegion

                return FMoWRegion(self.path, download=True)
            case "fmowyear":
                from .dg import FMoWYear

                return FMoWYear(self.path, download=True)
            case "officehome":
                from .dg import OfficeHome

                return OfficeHome(self.path, download=True)
            case "pacs":
                from .dg import PACS

                return PACS(self.path, download=True)
            case "rotatedmnist":
                from .dg import RotatedMNIST

                return RotatedMNIST(self.path, download=True)
            case "terraincognita":
                from .dg import TerraIncognita

                return TerraIncognita(self.path, download=True)
            case "vlcs":
                from .dg import VLCS

                return VLCS(self.path, download=True)
            case "povertymapurbanicity":
                from .dg import PovertyMapUrbanicity

                return PovertyMapUrbanicity(self.path, download=True)
            case "povertymapcountry":
                from .dg import PovertyMapCountry

                return PovertyMapCountry(self.path, download=True)
            case other:
                raise NotImplementedError(f"Unrecognized dataset: {other}")

    def _get_splits(self, dataset, seed):
        if self.domain_split == "id":
            domains = dataset.all_domains[:-1]
        elif self.domain_split == "ood":
            domains = dataset.all_domains[-1:]
        else:
            raise ValueError(f"Expected domain_split to be 'id' or 'ood', got {self.domain_split}.")

        # get the full domains
        domains = [dataset.domain(d) for d in domains]

        if self.domain_split != "ood":
            # get the seeded random splits for the given domain
            split = list(self.splits.keys()).index(self.split)
            generator = torch.Generator().manual_seed(seed)
            domains = [random_split(d, list(self.splits.values()), generator=generator)[split] for d in domains]

        return domains

    def _get_dataset(self, dataset, seed=0):
        raw_dataset = self._get_raw_dataset(dataset)
        splits = self._get_splits(raw_dataset, seed)

        return MultiDomainDataset(
            *splits, in_distribution=self.domain_split == "id", n_domains=len(raw_dataset.all_domains)
        )
