from collections.abc import Sequence, Callable

import numpy as np


from torch.utils.data import Dataset


class OxelDataset(Dataset):
    def __init__(self, files: Sequence[str], augmentation: Callable):
        self.files = files
        self.augmenation = augmentation
        self.indices = np.arange(len(files))

    def __len__(self):
        return len(self.files)

    def shuffle(self):
        self.indices = np.random.choice(len(self.files), len(self.files), replace=False)

    def __getitem__(self, index):
        file = self.files[self.indices[index]]

        view1, view2, indices, flags, mask1, mask2 = self._get_data(file)

        view1 = self.augmenation(view1).transpose((2, 0, 1))
        view2 = self.augmenation(view2).transpose((2, 0, 1))

        return view1, view2, indices, flags, mask1, mask2
