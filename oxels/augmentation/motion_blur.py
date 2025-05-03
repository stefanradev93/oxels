import numpy as np
import cv2

from scipy.ndimage import gaussian_filter


class MotionBlur:
    """
    Applies a random motion blur to an image using a simulated motion kernel.

    Parameters
    ----------
    L_min : int, optional
        Minimum length of the random motion trajectory. Default is 4.
    L_max : int, optional
        Maximum length of the random motion trajectory. Default is 33.
    s_min : int, optional
        Minimum standard deviation for Gaussian smoothing of the motion kernel. Default is 1.
    s_max : int, optional
        Maximum standard deviation for Gaussian smoothing of the motion kernel. Default is 11.

    Notes
    -----
    The motion kernel is generated from a random walk trajectory, then smoothed with a Gaussian filter.
    """

    def __init__(self, L_min: int = 4, L_max: int = 33, s_min: int = 1, s_max: int = 11):
        self.L_min = L_min
        self.L_max = L_max
        self.s_min = s_min
        self.s_max = s_max

    def get_motion_kernel(self):
        """
        Generate a random motion blur kernel based on a 2D random trajectory.

        Returns
        -------
        np.ndarray
            A normalized 2D motion blur kernel of shape (H, W).
        """
        L = np.random.randint(self.L_min, self.L_max)
        x = np.cumsum(np.cumsum(np.random.normal(0, 1, L)))
        y = np.cumsum(np.cumsum(np.random.normal(0, 1, L)))

        x -= np.min(x)
        y -= np.min(y)
        wf = int(np.max(x)) + 1
        hf = int(np.max(y)) + 1

        sigma = np.random.randint(low=self.s_min, high=self.s_max + 1)
        F = np.zeros((hf + 10 * sigma, wf + 10 * sigma))

        F[y.astype(int) + 5 * sigma, x.astype(int) + 5 * sigma] = 1
        K = cv2.resize(gaussian_filter(F, sigma), dsize=(0, 0), fx=0.1, fy=0.1)

        X, Y = np.meshgrid(np.arange(K.shape[1]), np.arange(K.shape[0]))
        cx = np.sum(K * X) / np.sum(K)
        cy = np.sum(K * Y) / np.sum(K)

        py = int(np.round(K.shape[0] / 2 - cy))
        px = int(np.round(K.shape[1] / 2 - cx))

        return np.pad(K, ((max(0, py), max(0, -py)), (max(0, px), max(0, -px)))) / np.sum(K)

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """
        Apply a randomly generated motion blur to the input image.

        Parameters
        ----------
        im : np.ndarray
            Input image of shape (H, W, C) or (H, W). Should be a NumPy array.

        Returns
        -------
        np.ndarray
            Blurred image of the same shape as the input.
        """
        K = self.get_motion_kernel()
        return cv2.filter2D(im, -1, K)
