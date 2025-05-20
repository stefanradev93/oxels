
from collections.abc import Callable
from typing import overload, TypeVar

import torch.distributed as dist


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


@allow_args
def rank_zero(fn, verbose=False, error=False):
    """
    Decorator to run a function only on rank 0.
    """
    def wrapper(*args, **kwargs):
        rank = dist.get_rank()
        if rank != 0:
            if error:
                raise RuntimeError(f"Function {fn.__name__} should only be called by rank 0")
            if verbose:
                print(f"Rank {rank} is skipping {fn.__name__}")
            return None
        if verbose:
            print(f"Rank {rank} is running {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper
