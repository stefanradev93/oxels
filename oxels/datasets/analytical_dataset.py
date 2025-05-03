from collections.abc import Sequence, Callable
import numpy as np

from .oxel_dataset import OxelDataset
from oxels.analytical_benchmark import TwoShapes
from oxels.transforms import ImagePerspectiveTransform


class AnalyticalDataset(OxelDataset):
    def __init__(self, n_samples: int, augmentation: Callable, w, h, frac_keep=0.125):
        super().__init__([None]*n_samples, augmentation) #we're not using files
        self.two_shapes = TwoShapes(w, h)

    def _get_data(self, file: str) -> Sequence[np.ndarray]:
        """
        Returns

        Parameters
        ----------
        file : ignored
        
        Returns
        -------
        view1   : ndarray, shape (H, W, 3), dtype=float
            First view
        view2   : ndarray, shape (H, W, 3), dtype=float
            Second view
        indices : ndarray, shape (H*W), dtype=int
            Indices from 0 to H*W-1 of view2.
        flags   : ndarray, shape (H*W), dtype=bool
            Flags to that indicate correct matches between view1 and transformed view2:
            flags[i] is true if view2[indices[i]//W, indices[i]%W] is (nearly) the same as view1[i//w, i%w]
        mask1   : ndarray, shape (H, W), dtype=bool
            Which pixels of view1 should be used for loss computation
        mask2   : ndarray, shape (H, W), dtype=bool
            Which pixels of view2 should be used for loss computation
        """
        
        im = self.two_shapes.get_image

        view1, view2, indices, flags, _, _ = self.transform.get_views_and_permutation(im)
        mask = np.ones(flags.shape, dtype=bool)

        return view1, view2, indices, flags, mask, mask