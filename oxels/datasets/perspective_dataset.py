from collections.abc import Sequence, Callable

import cv2
import numpy as np

from .oxel_dataset import OxelDataset
from oxels.transforms import ImagePerspectiveTransform


class PerspectiveDataset(OxelDataset):
    def __init__(self, files: Sequence[str], augmentation: Callable, w, h, frac_keep=0.125):
        super().__init__(files, augmentation)
        self.transform = ImagePerspectiveTransform(w, h, frac_keep=frac_keep)

    def _get_data(self, file: str) -> Sequence[np.ndarray]:
        """
        Returns

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
            Flags to that indicate correct matches between view1 and transformed view2:
            flags[i] is true if view2[indices[i]//W, indices[i]%W] is (nearly) the same as view1[i//w, i%w]
        mask1   : ndarray, shape (H, W), dtype=bool
            Which pixels of view1 should be used for loss computation
        mask2   : ndarray, shape (H, W), dtype=bool
            Which pixels of view2 should be used for loss computation
        """
        
        img_bgr = cv2.imread(file, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0

        view1, view2, indices, flags, mask1, mask2 = self.transform.get_views_and_permutation(img_rgb)

        return view1, view2, indices, flags, mask1, mask2
