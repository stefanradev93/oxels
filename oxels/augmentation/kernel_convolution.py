import numpy as np
import cv2


class KernelConvolution:
    """
    Applies a random convolutional kernel to an image.

    Parameters
    ----------
    max_size : int, optional
        Maximum size for the kernel in each dimension. The actual kernel size
        is randomly sampled as (h, w), where h, w ∈ [1, max_size]. Default is 3.

    Notes
    -----
    - The kernel is sampled from integer values in [-2, 2].
    - If the sampled kernel is all zeros, the image is returned unchanged.
    - The image is re-normalized to its original min/max range after filtering.
    """

    def __init__(self, max_size: int = 3):
        self.max_size = max_size

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """
        Apply a randomly generated kernel to the input image.

        Parameters
        ----------
        im : np.ndarray
            Input image of shape (H, W, C) or (H, W), with values in any range.

        Returns
        -------
        np.ndarray
            Output image of the same shape, with the original value range restored.
        """
        immin, immax = im.min(), im.max()
        sizex, sizey = np.random.randint(1, self.max_size + 1, size=2)

        kernel = np.random.randint(-2, 3, size=(sizex, sizey)).astype(float)

        if np.sum(np.abs(kernel)) == 0:
            return im  # Avoid dividing by zero

        # Normalize kernel
        kernel /= np.sum(np.abs(kernel))

        # Apply convolution
        im = cv2.filter2D(im, -1, kernel)

        # Normalize back to original range
        im -= im.min()
        im /= np.abs(im.max()) + 1e-6
        im *= immax - immin
        im += immin
        return im
