import numpy as np
import cv2


class HueShift:
    """
    Applies a random linear gradient (phase shift) to the hue channel of an RGB image.

    This augmentation simulates smooth, spatially varying color shifts by applying a 2D linear gradient
    to the hue channel in HSV space. The gradient is randomly sampled in both x and y directions,
    as well as a global offset.

    Parameters
    ----------
    max_shift : float, optional
        Maximum hue shift as a fraction of the full hue range. A value of 1.0 corresponds to
        a full rotation around the hue circle. Default is 0.5.

    Notes
    -----
    - Input image must be in RGB format with float values in [0, 1].
    - The operation is performed in HSV color space and only affects the hue channel.
    - Output is in the same RGB float format, normalized to [0, 1].

    References
    ----------
    Hue channel in OpenCV HSV format ranges from 0 to 179 (not 0 to 360).
    """

    def __init__(self, max_shift: float = 0.5):
        self.max_shift = max_shift

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a random hue gradient shift to an RGB image.

        Parameters
        ----------
        image : np.ndarray
            Input RGB image of shape (H, W, 3), with float values in [0, 1].

        Returns
        -------
        np.ndarray
            RGB image of shape (H, W, 3), dtype float32, with hue shifted.
        """
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        h_norm = h / 179.0

        H, W = h.shape
        intercept = np.random.uniform(-self.max_shift, self.max_shift)
        slope_x = np.random.uniform(-self.max_shift, self.max_shift)
        slope_y = np.random.uniform(-self.max_shift, self.max_shift)

        xv = np.linspace(0, 1, W, dtype=np.float32)
        yv = np.linspace(0, 1, H, dtype=np.float32)
        grad = intercept + slope_x * xv[np.newaxis, :] + slope_y * yv[:, np.newaxis]

        h_shifted = (h_norm + grad) % 1.0
        h_new = (h_shifted * 179.0).astype(np.float32)

        hsv[..., 0] = h_new
        hsv[..., 1] = s
        hsv[..., 2] = v

        out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out_rgb / 255.0
