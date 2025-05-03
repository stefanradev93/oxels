import numpy as np
import cv2


class PerspectiveTransform:
    """
    Generates a pair of spatially transformed views of an image using a random perspective
    transformation, and returns the index mapping between them.

    This is useful for contrastive self-supervised learning or patch matching, where geometric
    variation is required while preserving local correspondences.

    Parameters
    ----------
    w : int
        Width of the output views.
    h : int
        Height of the output views.
    frac_keep : float, optional
        Fraction of pixels for which correspondence is preserved. Default is 0.125.
    max_shift : int, optional
        Maximum pixel shift for translation. If None, defaults to h // 4.
    std_matrix_noise : float, optional
        Standard deviation of Gaussian noise added to the perspective matrix. Default is 0.1.

    Attributes
    ----------
    X : np.ndarray
        Precomputed x-coordinate meshgrid of shape (h, w).
    Y : np.ndarray
        Precomputed y-coordinate meshgrid of shape (h, w).
    """

    def __init__(
        self,
        w: int,
        h: int,
        frac_keep: float = 0.125,
        max_shift: int = None,
        std_matrix_noise: float = 0.1
    ):
        self.w = int(w)
        self.h = int(h)
        self.frac_keep = frac_keep

        if max_shift is None:
            max_shift = h // 4

        self.max_shift = max_shift
        self.std_matrix_noise = std_matrix_noise

        self.X, self.Y = np.meshgrid(np.arange(w), np.arange(h))

    def get_index_permutation(self, H: np.ndarray, sub_x: int, sub_y: int, mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the pixel index permutation under a perspective transformation.

        Parameters
        ----------
        H : np.ndarray
            3x3 perspective transformation matrix.
        sub_x : int
            Horizontal translation shift.
        sub_y : int
            Vertical translation shift.
        mask : np.ndarray, optional
            Optional binary mask (H, W) specifying invalid pixels. Default is None.

        Returns
        -------
        ind_flat : np.ndarray
            Flattened indices (1D) mapping each pixel to its corresponding pixel after transformation.
        match : np.ndarray
            Boolean array of shape (H*W,) indicating valid matches.
        """
        H_inv = np.linalg.inv(H)
        l = 1 / (H_inv[2, 0] * self.X + H_inv[2, 1] * self.Y + H_inv[2, 2])
        X_ = (-H_inv[0, 0] * H[0, 2] - H_inv[0, 1] * H[1, 2] + H_inv[0, 0] * self.X + H_inv[0, 1] * self.Y) * l
        Y_ = (-H_inv[1, 0] * H[0, 2] - H_inv[1, 1] * H[1, 2] + H_inv[1, 0] * self.X + H_inv[1, 1] * self.Y) * l

        ind = np.round(np.stack([Y_.ravel(), X_.ravel()], axis=1)).astype(int)
        match = ((ind[:, 1] > sub_x) & (ind[:, 1] < self.w + sub_x) &
                 (ind[:, 0] > sub_y) & (ind[:, 0] < self.h + sub_y)).reshape((-1, 1))
        match &= (np.random.random(match.shape) < self.frac_keep)

        random_ind = np.stack([
            np.random.randint(0, self.h, self.w * self.h),
            np.random.randint(0, self.w, self.w * self.h)
        ], axis=1)

        ind = match * (ind - [[sub_y, sub_x]]) + (~match) * random_ind

        if mask is not None:
            mask_flat = mask.ravel()
            valid = mask_flat[ind[:, 0] * self.w + ind[:, 1]].reshape(match.shape)
            match &= (valid == 0)
            ind = match * ind + (~match) * random_ind

        ind_flat = ind[:, 0] * self.w + ind[:, 1]
        return ind_flat, match.ravel()

    def get_views_and_permutation(self, im: np.ndarray):
        """
        Applies a random perspective transformation to generate a pair of spatially
        related image views, along with the index permutation between them.

        Parameters
        ----------
        im : np.ndarray
            Input image of shape (H, W, C) or (H, W).

        Returns
        -------
        view1 : np.ndarray
            First view obtained via perspective warping.
        view2 : np.ndarray
            Second view, a translated crop from the original image.
        permutation : np.ndarray
            Flattened index permutation mapping view1 to view2.
        flags : np.ndarray
            Boolean mask indicating which pixels in the permutation are valid.
        mask1 : np.ndarray
            Boolean flat mask for valid pixels in view1.
        mask2 : np.ndarray
            Boolean flat mask for valid pixels in view2.
        """
        if im.shape[0] <= self.h + 2 * self.max_shift or im.shape[1] <= self.w + 2 * self.max_shift:
            im = cv2.resize(
                im,
                (max(im.shape[1], self.w + 2 * self.max_shift + 1),
                 max(im.shape[0], self.h + 2 * self.max_shift + 1))
            )

        x = np.random.randint(self.max_shift + self.w // 2, im.shape[1] - self.max_shift - self.w // 2 + 1)
        y = np.random.randint(self.max_shift + self.h // 2, im.shape[0] - self.max_shift - self.h // 2 + 1)

        alpha = np.random.uniform(0, 2 * np.pi)
        dx, dy = np.random.randint(-self.max_shift, self.max_shift + 1, size=2)

        B = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        B += np.random.normal(0, self.std_matrix_noise, size=(2, 2))

        A = np.eye(3, dtype=np.float32)
        A[:2, :2] = B
        A[:2, 2] = -A[:2, :2].dot([self.w // 2, self.h // 2]) + [self.w // 2, self.h // 2]

        permutation, flags = self.get_index_permutation(A, dx, dy)

        A[:2, 2] = -A[:2, :2].dot([x, y]) + [self.w // 2, self.h // 2]

        view1 = cv2.warpPerspective(im, A, (self.w, self.h), flags=cv2.INTER_AREA)
        view2 = im[y + dy - self.h // 2 : y + dy + self.h // 2,
                   x + dx - self.w // 2 : x + dx + self.w // 2]

        valid = np.ones(im.shape[:2], dtype=np.uint8)
        mask1 = cv2.warpPerspective(valid, A, (self.w, self.h), flags=cv2.INTER_AREA).ravel() == 1
        mask2 = np.ones(self.h * self.w, dtype=bool)

        return view1, view2, permutation, flags, mask1, mask2
