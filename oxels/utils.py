import torch
import torch.nn.functional as F

import hashlib
from pathlib import Path


def md5(filepath, chunk_size=1024 ** 2):
    """Returns the MD5 hash of a given file"""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(filepath)

    with open(filepath, "rb") as f:
        return hashlib.file_digest(f, "md5", _bufsize=128 * chunk_size).hexdigest()


def pad_to_even_height_and_width(x: torch.Tensor, padding_value: float = 0.0) -> torch.Tensor:
    *_, channels, height, width = x.shape

    padding_left = 0
    padding_right = width % 2
    padding_top = 0
    padding_bottom = height % 2

    padding = [padding_left, padding_right, padding_top, padding_bottom]

    return F.pad(x, padding, mode="constant", value=padding_value)
