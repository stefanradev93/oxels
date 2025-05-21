import numpy as np


def encode_floats_to_bits(arr: np.ndarray, max_range: float, n_bits: int) -> np.ndarray:
    """
    Quantize a 2D float array into n_bits and return its bit‑plane encoding as booleans.

    Parameters
    ----------
    arr : np.ndarray
        2D array of floats, each 0 <= arr[i,j] <= max_range.
    max_range : float
        Maximum possible value in `arr`. Values outside [0, max_range] will error.
    n_bits : int
        Number of bits for encoding (e.g. 8 for 0–255 resolution).

    Returns
    -------
    bits : np.ndarray
        Boolean array of shape (H, W, n_bits).  bits[i,j,k] is the k-th bit
        (MSB at k=0) of the quantized arr[i,j].
    """
    if arr.ndim != 2:
        raise ValueError("`arr` must be 2D")
    if np.any(arr < 0) or np.any(arr > max_range):
        raise ValueError("All array values must lie within [0, max_range]")

    # 1) scale & quantize into unsigned integers 0 … (2**n_bits - 1)
    max_int = (1 << n_bits) - 1
    scale = max_int / max_range
    # round to nearest integer
    ints = np.floor(arr * scale + 0.5).astype(np.uint32)

    # 2) unpack bits: create bitmask shifts [n_bits-1 … 0]
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.uint32)
    # broadcasting: (H,W,1) >> (n_bits,)  →  (H,W,n_bits)
    bits = ((ints[..., None] >> shifts) & 1).astype(bool)

    return bits


class TwoShapes:
    def __init__(self, w=64, h=64):
        self.w = w
        self.h = h
        self.X, self.Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

    def dist2_to_segment(self, xy1, xy2):
        T = (
            1
            / np.linalg.norm(xy1 - xy2) ** 2
            * ((self.X - xy1[0]) * (xy2[0] - xy1[0]) + (self.Y - xy1[1]) * (xy2[1] - xy1[1]))
        )
        XC = (T < 0) * xy1[0] + (T >= 0) * (T < 1) * (xy1[0] + T * (xy2[0] - xy1[0])) + (T > 1) * xy2[0]
        YC = (T < 0) * xy1[1] + (T >= 0) * (T < 1) * (xy1[1] + T * (xy2[1] - xy1[1])) + (T > 1) * xy2[1]

        return (XC - self.X) ** 2 + (YC - self.Y) ** 2

    def get_image(self, cs, Ms, types, cg, Mg):
        Ps = []
        for i in range(len(cs)):
            c = cg + Mg.dot(cs[i])
            M = Mg.dot(Ms[i])

            if types[i]:
                x0y0 = c - M.dot([0.0, 0.125])
                x1y0 = c + M.dot([0.125, 0])
                x0y1 = c + M.dot([0, 0.125])

                d2_1 = self.dist2_to_segment(c, x1y0)
                d2_2 = self.dist2_to_segment(x0y0, x0y1)
                d2 = (d2_1 < d2_2) * d2_1 + (d2_1 >= d2_2) * d2_2

                Ps.append(0.01 / (0.01 + d2))
            else:
                x1y0 = c + M.dot([0.125, 0])
                x0y1 = c + M.dot([0, 0.125])

                d2_1 = self.dist2_to_segment(c, x1y0)
                d2_2 = self.dist2_to_segment(c, x0y1)
                d2 = (d2_1 < d2_2) * d2_1 + (d2_1 >= d2_2) * d2_2
                Ps.append(0.01 / (0.01 + d2))

        return np.max(Ps, axis=0)

    def get_polar_coordinates(self, cs, Ms, types, cg, Mg):
        cs = np.reshape(cg, (1, 2)) + np.dot(cs, Mg)
        M_invs = np.linalg.inv(np.matmul(Ms, Mg))
        types = np.array(types).reshape((-1, 1, 1))

        image_R2s = [
            (self.X - c[0]) ** 2 + (self.Y - c[1]) ** 2 + np.random.random((self.h, self.w)) * 1e-8 for c in cs
        ]  # random for tie breaking
        min_R2 = np.min(image_R2s, axis=0)
        flags_R2 = np.array([R2 == min_R2 for R2 in image_R2s])

        local_Xs = np.array(
            [(self.X - cs[i, 0]) * M_invs[i, 0, 0] + (self.Y - cs[i, 1]) * M_invs[i, 0, 1] for i in range(len(cs))]
        )
        local_Ys = np.array(
            [(self.X - cs[i, 0]) * M_invs[i, 1, 0] + (self.Y - cs[i, 1]) * M_invs[i, 1, 1] for i in range(len(cs))]
        )

        local_Rs = np.hypot(local_Ys, local_Xs)
        local_As = np.arctan2(local_Ys, local_Xs)
        return (
            np.sum(local_Rs * flags_R2, axis=0),
            np.sum(local_As * flags_R2, axis=0),
            np.sum(types * flags_R2, axis=0),
        )

    def get_encoded_coordinates(self, cs, Ms, types, cg, Mg, n_bits_dist, n_bits_angle, max_dist):
        local_Rs, local_As, local_types = self.get_polar_coordinates(cs, Ms, types, cg, Mg)
        n_bits_type = 1

        oxels = np.zeros((self.h, self.w, n_bits_type + n_bits_dist + n_bits_angle))
        oxels[:, :, :n_bits_type] = encode_floats_to_bits(local_types, 2, n_bits_type)
        local_Rs = np.clip(local_Rs, 0, max_dist)
        oxels[:, :, n_bits_type : n_bits_type + n_bits_dist] = encode_floats_to_bits(local_Rs, max_dist, n_bits_dist)
        local_As = np.clip(local_As + np.pi, 0, 2 * np.pi)
        oxels[:, :, n_bits_type + n_bits_dist :] = encode_floats_to_bits(local_As, 2 * np.pi, n_bits_angle)

        return oxels
