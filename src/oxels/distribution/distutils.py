
import torch.distributed as dist

from oxels.utils import allow_args


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
