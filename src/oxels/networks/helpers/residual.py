from collections.abc import Callable, Sequence

import torch.nn as nn


class Residual(nn.Module):
    def __init__(
        self,
        *fns: Callable | Sequence[Callable],
        in_features: int,
        out_features: int,
    ):
        super().__init__()

        if len(fns) == 1 and isinstance(fns[0], Sequence):
            # support both Residual(fn, ...) and Residual([fn, ...])
            fns = fns[0]

        self.inner = nn.Sequential(*fns)

        if in_features == out_features:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Linear(in_features, out_features, bias=False)
            nn.init.orthogonal_(self.projector.weight)

        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x, **kwargs):
        return self.projector(x) + self.inner(x, **kwargs)
