import numpy as np
import cv2


class Sobel:
    """
    Applies the Sobel operator to compute the gradient magnitude of an image.

    Notes
    -----
    - Uses OpenCV's `cv2.Sobel` function to compute image gradients along the x and y axes.
    - The output is the L2 norm (Euclidean magnitude) of the gradient at each pixel.
    - Input should be a grayscale or single-channel image. For color images, apply per channel if needed.

    References
    ----------
    OpenCV Sobel operator:
    https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    """

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """
        Compute the Sobel gradient magnitude of the input image.

        Parameters
        ----------
        im : np.ndarray
            Input image of shape (H, W) or (H, W, C), with values in any range.

        Returns
        -------
        np.ndarray
            Gradient magnitude image of shape (H, W), dtype float32.
        """
        grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1)
        grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0)
        return np.sqrt(grad_x**2 + grad_y**2)
