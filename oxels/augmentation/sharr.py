import numpy as np
import cv2


class Scharr:
    """
    Applies the Scharr operator to compute the gradient magnitude of an image.

    Notes
    -----
    - Uses OpenCV's `cv2.Scharr` to compute image gradients along the x and y directions.
    - The output is the L2 norm (Euclidean magnitude) of the gradient at each pixel.
    - Input image should be a single-channel or 3-channel image with dtype convertible to float.

    References
    ----------
    OpenCV Scharr operator:
    https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    """

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """
        Compute the Scharr gradient magnitude of the input image.

        Parameters
        ----------
        im : np.ndarray
            Input image array of shape (H, W) or (H, W, C).

        Returns
        -------
        np.ndarray
            Gradient magnitude image of the same spatial dimensions as the input,
            with float32 dtype.
        """
        grad_y = cv2.Scharr(im, cv2.CV_32F, 0, 1)
        grad_x = cv2.Scharr(im, cv2.CV_32F, 1, 0)
        return np.sqrt(grad_x**2 + grad_y**2)
