from collections.abc import Sequence

import torch.nn as nn

from typing import Optional

from oxels.typing import ActivationType


class ConvBlock(nn.Sequential):
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
        use_batchnorm: bool = True,
        dropout: Optional[float] = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        features = [in_features, *hidden_features]
        for in_features, out_features in zip(features[:-1], features[1:]):
            self.append(nn.Conv2d(in_features, out_features, kernel_size, stride, padding="same"))
            self.append(activation())

            if use_batchnorm:
                self.append(nn.BatchNorm2d(out_features))

            if dropout is not None and dropout > 0.0:
                self.append(nn.Dropout2d(dropout))

        self.append(nn.Conv2d(features[-1], self.out_features, kernel_size, stride, padding="same"))

        if out_activation is not None:
            self.append(out_activation())
