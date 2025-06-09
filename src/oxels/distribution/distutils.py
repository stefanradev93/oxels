import torch.distributed as dist

import os

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from oxels.utils import allow_args

T = TypeVar("T")


def count_nodes() -> int:
    return int(os.environ.get("SLURM_NNODES", 1))


def get_local_rank() -> int:
    return int(os.environ.get("SLURM_PROCID", 0)) % int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))


def get_rank() -> int:
    return int(os.environ.get("SLURM_PROCID", 0))


def get_world_size() -> int:
    return int(os.environ.get("SLURM_NTASKS", 1))


def get_job_id() -> int:
    return int(os.environ.get("SLURM_JOB_ID", 0))


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


def recv_object(src: int = 0, group: dist.ProcessGroup = None) -> any:
    object_list = [None]
    dist.recv_object_list(object_list, src=src, group=group)
    return object_list[0]


def send_object(obj: any, src: int = None, dst: int | Sequence[int] = None, group: dist.ProcessGroup = None) -> None:
    if dst is None:
        if src is None:
            src = dist.get_rank(group)

        dst = list(range(dist.get_world_size(group)))
        dst.remove(src)

    object_list = [obj]
    for rank in dst:
        dist.send_object_list(object_list, dst=rank)


def send_or_recv(
    fn: Callable[[], T], src_dst: Mapping[int, int | Sequence[int]] = None, group: dist.ProcessGroup = None
) -> T:
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)

    if src_dst is None:
        src_dst = {0: list(range(1, size))}
    else:
        # ensure the mapping is unique and complete
        seen_ranks = set()
        for key, value in src_dst.items():
            for _rank in [key, *value]:
                if _rank in seen_ranks:
                    raise ValueError(f"src_dst mapping must be unique, but rank {key} appears more than once.")

                seen_ranks.add(_rank)

        missing_ranks = seen_ranks - set(range(size))
        if missing_ranks:
            raise ValueError(f"src_dst mapping must cover all ranks, but is missing {list(missing_ranks)}")

    dst_src = {}
    for src, value in src_dst.items():
        for dst in value:
            dst_src[dst] = src

    if rank in src_dst:
        obj = fn()
        dst = src_dst[rank]

        send_object(obj, dst=dst, group=group)
    else:
        src = dst_src[rank]

        obj = recv_object(src=src, group=group)

    return obj


def all_try(fn: Callable[[], T], group: dist.ProcessGroup = None) -> (Sequence[T | None], Sequence[Exception | None]):
    try:
        # TODO: this assumes no blocking distributed operations are in fn, like send_or_recv
        #  can we change this? It's problematic if some ranks are spinning while others have raised an exception
        value = fn()
        error = None
    except Exception as e:
        value = None
        error = e

    size = dist.get_world_size(group)

    results = {}
    for i in range(size):
        src = i
        dst = list(range(size))
        dst.remove(src)

        # TODO: could just use all_gather_object
        results[i] = send_or_recv(lambda: (value, error), src_dst={src: dst}, group=group)

    values = [v[0] for v in results.values()]
    errors = [v[1] for v in results.values()]

    return values, errors


def call_once(fn: Callable[[], any], group: dist.ProcessGroup = None) -> None:
    rank = dist.get_rank(group)
    if rank == 0:
        fn()
