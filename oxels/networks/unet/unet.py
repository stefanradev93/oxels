from collections.abc import Sequence
from typing import Optional

import torch.nn as nn
from torchvision.transforms.functional import center_crop

from oxels.typing import ActivationType, UNetBlockType as BlockType
from oxels.utils import pad_to_even_height_and_width

from ..helpers import ConvBlock


class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        down_features: Sequence[Sequence[int]],
        middle_features: Sequence[int],
        up_features: Sequence[Sequence[int]],
        activation: ActivationType = nn.SiLU,
        out_activation: Optional[ActivationType] = None,
        down_types: BlockType | Sequence[BlockType] = ConvBlock,
        middle_type: BlockType = ConvBlock,
        up_types: BlockType | Sequence[BlockType] = ConvBlock,
    ):
        super().__init__()

        if len(down_features) != len(up_features):
            raise ValueError(
                f"Need the same number of stages in `down_features` and `up_features`, "
                f"but got {len(down_features)} and {len(up_features)}."
            )

        if not isinstance(down_types, Sequence):
            down_types = [down_types for _ in range(len(down_features))]
        if not isinstance(up_types, Sequence):
            up_types = [up_types for _ in range(len(up_features))]

        down_type = down_types[0]
        up_type = up_types[-1]

        self.in_features = in_features
        self.out_features = out_features
        # avoid bottlenecks
        hidden_features = max(in_features, out_features)
        self.hidden_features = hidden_features

        self.encoder = down_type(
            down_features[0],
            in_features=in_features,
            out_features=hidden_features,
            activation=activation,
        )
        self.encoder_pooling = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            2,
            padding=1,
        )

        if len(down_features) > 1:
            # construct the next level as a recursive UNet
            self.middle = UNet(
                in_features=hidden_features,
                out_features=hidden_features,
                down_features=down_features[1:],
                middle_features=middle_features,
                up_features=up_features[:-1],
                activation=activation,
                down_types=down_types[1:],
                middle_type=middle_type,
                up_types=up_types[:-1],
            )
        else:
            self.middle = middle_type(
                middle_features, in_features=hidden_features, out_features=hidden_features, activation=activation
            )

        self.decoder = up_type(
            up_features[-1],
            in_features=hidden_features,
            out_features=out_features,
            activation=activation,
        )
        self.decoder_pooling = nn.ConvTranspose2d(
            hidden_features,
            hidden_features,
            3,
            2,
            padding=1,
        )

        if in_features != out_features:
            self.projector = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding="same", bias=False)
            nn.init.orthogonal_(self.projector.weight)
        else:
            self.projector = nn.Identity()

        if out_activation is not None:
            self.out_activation = out_activation()
        else:
            self.out_activation = nn.Identity()

    def __call__(self, x, **kwargs):
        # pad so we don't lose information on the down-pooling
        x = pad_to_even_height_and_width(x)
        residual = self.projector(x)

        x = self.encoder(x, **kwargs)
        x = self.middle(x, **kwargs)
        x = self.decoder(x, **kwargs)

        # center-crop after the forward pass to discard unusable pixels
        x = center_crop(x, list(residual.shape[-2:]))

        x = x + residual

        return self.out_activation(x)
