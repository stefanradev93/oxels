import torch
import torch.nn.functional as F

import hashlib
from pathlib import Path


from collections.abc import Callable
from typing import overload, TypeVar


Fn = TypeVar("Fn", bound=Callable[..., any])

# this can be done better, but not compactly in Python < 3.12
Decorator = Fn


def allow_args(fn: Decorator) -> Decorator:
    """Decorator to allow another decorator to be called with or without arguments."""

    @overload
    def wrapper(f: Fn) -> Fn: ...
    @overload
    def wrapper(*fargs: any, **fkwargs: any) -> Fn: ...
    def wrapper(*fargs: any, **fkwargs: any) -> Fn:
        if len(fargs) == 1 and not fkwargs and callable(fargs[0]):
            # called without arguments
            return fn(fargs[0])
        else:
            # called with arguments, bind
            return lambda f: fn(f, *fargs, **fkwargs)

    return wrapper


def md5(filepath, chunk_size=1024**2):
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
