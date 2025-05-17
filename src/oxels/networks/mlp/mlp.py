from collections.abc import Sequence

import torch.nn as nn

from ..helpers import Residual


class MLP(nn.Sequential):
    def __init__(
        self,
        features: Sequence[int],
        *,
        in_features: int,
        out_features: int,
        activation=nn.Mish,
        out_activation=None,
        residual=True,
        dropout=0.05,
        norm=None,
    ):
        features = [in_features, *features]

        layers = []
        for _in_features, _out_features in zip(features[:-1], features[1:]):
            layer = self._make_layer(
                _in_features,
                _out_features,
                activation,
                residual,
                dropout,
                norm,
            )
            layers.append(layer)

        layers.append(nn.Linear(features[-1], out_features))

        if out_activation is not None:
            layers.append(out_activation)

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features

    @staticmethod
    def _make_layer(
        in_features,
        out_features,
        activation,
        residual,
        dropout,
        norm,
    ):
        layers = []

        linear = nn.Linear(
            in_features,
            out_features,
        )
        layers.append(linear)

        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(activation)

        if norm == "batch":
            layers.append(nn.BatchNorm1d(out_features))
        elif norm == "layer":
            layers.append(nn.LayerNorm(out_features))
        elif isinstance(norm, str):
            raise ValueError(f"Unknown normalization strategy: {norm!r}.")
        elif isinstance(norm, nn.Module):
            layers.append(norm)
        elif norm is None:
            pass
        else:
            raise TypeError(f"Cannot infer norm from {norm!r} of type {type(norm)}.")

        if residual:
            return Residual(
                *layers,
                in_features=in_features,
                out_features=out_features,
            )

        return nn.Sequential(*layers)
