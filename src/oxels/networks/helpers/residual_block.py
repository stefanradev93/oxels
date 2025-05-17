from collections.abc import Sequence

import torch.nn as nn

from typing import Optional

from oxels.typing import ActivationType


class ResidualLayer(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        kernel_size: int,
        stride: int,
        use_batchnorm: bool = False,
        dropout: float = 0.05,
        activation: ActivationType = nn.SiLU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if in_features == out_features:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Conv2d(
                in_features, out_features, kernel_size=kernel_size, stride=stride, padding="same"
            )

        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding="same")

        if use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_features)
        else:
            self.batchnorm = nn.Identity()

        if dropout is not None and dropout > 0.0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = nn.Identity()

        if activation is not None:
            self.activation = activation()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        residual = self.projector(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = x + residual
        return x


class ResidualBlock(nn.Sequential):
    def __init__(
        self,
        hidden_features: Sequence[int],
        *,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: ActivationType = nn.SiLU,
        out_activation: Optional[ActivationType] = None,
        use_attention: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_heads: int = 4,
        use_batchnorm: bool = False,
        dropout: Optional[float] = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        input_layer = ResidualLayer(
            in_features=in_features,
            out_features=hidden_features[0],
            kernel_size=kernel_size,
            stride=stride,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            activation=activation,
        )
        self.append(input_layer)

        for in_features, out_features in zip(hidden_features[:-1], hidden_features[1:]):
            hidden_layer = ResidualLayer(
                in_features=in_features,
                out_features=out_features,
                kernel_size=kernel_size,
                stride=stride,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
                activation=activation,
            )
            self.append(hidden_layer)

        output_layer = ResidualLayer(
            in_features=hidden_features[-1],
            out_features=self.out_features,
            kernel_size=kernel_size,
            stride=stride,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            activation=out_activation,
        )
        self.append(output_layer)
