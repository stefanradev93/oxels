from collections.abc import Sequence, Callable

import numpy as np


class SequentialStrategy:
    """
    Applies a sequence of image augmentations with optional probabilistic and ordering logic.

    Parameters
    ----------
    augmentations : Sequence[Callable]
        A list of callable augmentation functions, each taking an image (np.ndarray) and returning an augmented image.
        Could be a single augmentation or a strategy iteself.
    probabilities : Sequence[float], optional
        A list of probabilities corresponding to each augmentation. If provided, each augmentation is applied
        with its associated probability. If None, all augmentations are considered active.
    ordered : bool, optional
        If True, augmentations are applied in the given order. If False, the active augmentations are applied
        in random order. Default is False.
    at_most_one : bool, optional
        If True, only one randomly selected augmentation will be applied per call. Default is False.

    Notes
    -----
    `ordered` and `at_most_one` cannot both be True.
    """

    def __init__(
        self,
        augmentations: Sequence[Callable],
        probabilities: Sequence[float] = None,
        ordered: bool = False,
        at_most_one: bool = False,
        dtype: np.dtype = np.float32
    ):
        
        if ordered and at_most_one:
            raise ValueError("`ordered` and `at_most_one` cannot both be True.")
        
        if probabilities is not None:
            if len(probabilities) != len(augmentations):
                raise ValueError("`augmentations` and `probabilities` need to have the same length.")            
        
        self.augmentations = augmentations
        self.probabilities = probabilities
        self.ordered = ordered
        self.at_most_one = at_most_one
        self.dtype = dtype

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """
        Apply selected augmentations to the input image.

        Parameters
        ----------
        im : np.ndarray
            Input image of shape (H, W, C) or (H, W), depending on the augmentations used.

        Returns
        -------
        np.ndarray
            Augmented image.
        """
        if self.probabilities is None:
            active_indices = np.arange(len(self.augmentations))
        else:
            active_indices = np.where(np.random.rand(len(self.probabilities)) < self.probabilities)[0]

        if not self.ordered and len(active_indices):
            active_indices = active_indices[np.random.choice(len(active_indices), len(active_indices), replace=False)]

        for i in active_indices:
            im = self.augmentations[i](im)
            if self.at_most_one:
                break

        return im.astype(self.dtype, copy=False)
