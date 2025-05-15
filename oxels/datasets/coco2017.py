import torch
from torch.utils.data import Dataset
from pathlib import Path
from os import PathLike

from fiftyone.zoo.datasets import _parse_dataset_details

from oxels.transforms import PerspectiveTransform


class COCO2017(Dataset):
    default_path = Path(__file__).parents[2] / "datasets"
    default_path = default_path.resolve()

    def __init__(self, path: str | PathLike = default_path, w=256, h=256, frac_keep=0.125, split="train"):
        super().__init__()
        self.path = Path(path) / "COCO2017"
        self.dataset, _ = _parse_dataset_details("coco-2017")
        self.dataset.download_and_prepare(self.path, cleanup=True, split=split)

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


if __name__ == "__main__":
    dataset = COCO2017()
    print(len(dataset))
    img, *_ = dataset[0]

    print(img.shape, img.dtype, img.min(), img.max())
