from collections.abc import Sequence

import numpy as np


class BrightnessContrast:
    """
    Applies random brightness and contrast adjustment to an image.

    Parameters
    ----------
    contrast_range : tuple of float, optional
        Range (min, max) from which to sample the contrast scaling factor.
        A value >1 increases contrast, <1 decreases it. Default is (0.5, 1.5).
    brightness_range : tuple of float, optional
        Range (min, max) from which to sample the brightness shift.
        A value >0 brightens the image, <0 darkens it. Default is (-0.2, 0.2).

    Notes
    -----
    The image is assumed to have pixel values in [0, 1]. After applying the transformation,
    values are clipped to [0, 1] to preserve valid image intensities.
    """

    def __init__(
        self, 
        contrast_range: Sequence[float, float] = (0.5, 1.5), 
        brightness_range: Sequence[float, float] = (-0.2, 0.2)
    ):
        
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """
        Apply a random contrast and brightness adjustment to the image.

        Parameters
        ----------
        im : np.ndarray
            Input image of shape (H, W, C) or (H, W), with pixel values in [0, 1].

        Returns
        -------
        np.ndarray
            Adjusted image, clipped to the range [0, 1].
        """
        contrast = np.random.uniform(*self.contrast_range)
        brightness = np.random.uniform(*self.brightness_range)
        return np.clip(im * contrast + brightness, 0, 1)
