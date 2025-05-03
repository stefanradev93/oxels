import numpy as np


class PixelNoise:
    """
    Adds Gaussian noise to an image, optionally sampling the noise level from a range.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum value of the standard deviation for sampling noise. Only used if `sigma` is None.
    sigma_max : float, optional
        Maximum value of the standard deviation for sampling noise. Only used if `sigma` is None.

    Notes
    -----
    - `sigma` is sampled uniformly from [`sigma_min`, `sigma_max`] each time.
    - The resulting noisy image is clipped to [0, 1].
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 0.1):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to the image.

        Parameters
        ----------
        im : np.ndarray
            Input image array with values in [0, 1].

        Returns
        -------
        np.ndarray
            Noisy image array of the same shape, with values clipped to [0, 1].
        """
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        noise = np.random.normal(0, sigma, im.shape)
        return np.clip(im + noise, 0, 1)
