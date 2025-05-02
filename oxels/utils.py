
import torch
import torch.nn.functional as F


def pad_to_even_height_and_width(x: torch.Tensor, padding_value: float = 0.0) -> torch.Tensor:
    *_, channels, height, width = x.shape

    padding_left = 0
    padding_right = width % 2
    padding_top = 0
    padding_bottom = height % 2

    padding = [padding_left, padding_right, padding_top, padding_bottom]

    return F.pad(x, padding, mode="constant", value=padding_value)
