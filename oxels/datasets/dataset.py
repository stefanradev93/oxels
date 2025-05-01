import numpy as np
from collections.abc import Sequence, Callable

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

    def _get_data(self, file: str):
        """
        Parameters
        ----------
        file : filename
        

        Returns
        -------
        view1   : ndarray, shape (H, W, 3), dtype=float
            First view
        view2   : ndarray, shape (H, W, 3), dtype=float
            Second view
        indices : ndarray, shape (H*W), dtype=int
            Indices from 0 to H*W-1 of view2.
        flags   : ndarray, shape (H*W), dtype=bool
            Flags to that indicate correct matches between view1 and strasfomred view2:
            flags[i] is true if view2[indices[i]//W, indices[i]%W] is (nearly) the same as view1[i//w, i%w]
        mask1   : ndarray, shape (H*W), dtype=bool
            Which oxels of view1 should be used for loss computation
        mask2   : ndarray, shape (H*W), dtype=bool
            Which oxels of view2 should be used for loss computation
        """
        raise NotImplementedError
    
    def __getitem__(self, index):
        if index == 0:
            self.shuffle()

        file = self.files[self.indices[index]]

        view1, view2, indices, flags, mask1, mask2 = self._get_data(file)
        
        view1 = self.augmenation(view1).transpose((2,0,1))
        view2 = self.augmenation(view2).transpose((2,0,1))

        return view1, view2, indices, flags, mask1, mask2
