import torchvision.transforms as transforms

from pathlib import Path

from .mnist import MNISTDataset
from . import transforms as dtransforms


class ColoredMNIST(MNISTDataset):
    all_domains = ["L", "M", "H"]
    domain_map = {"L": 0, "M": 1, "H": 2}
    n_classes = 2

    def __init__(self, root: str | Path, download: bool = True, transform=None, domain_transforms=None):
        make_color_ch = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )
        if transform is None:
            transform = make_color_ch
        else:
            transform = transforms.Compose(make_color_ch.transforms + transform.transforms)

        if domain_transforms is None:
            domain_transforms = [
                dtransforms.Compose(
                    [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.25), dtransforms.ZeroChannel(0.1)]
                ),
                dtransforms.Compose(
                    [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.25), dtransforms.ZeroChannel(0.2)]
                ),
                dtransforms.Compose(
                    [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.25), dtransforms.ZeroChannel(0.9)]
                ),
            ]

        super().__init__(root, download=download, transform=transform, domain_transforms=domain_transforms)
