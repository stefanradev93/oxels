from collections.abc import Sequence

import torch.nn as nn
from torchvision.transforms.functional import center_crop

from oxels.utils import pad_to_even_height_and_width

from ..helpers import ConvBlock


class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        down_features: Sequence[Sequence[int]],
        up_features: Sequence[Sequence[int]],
        activation: type(nn.Module) = nn.SiLU,
    ):
        super().__init__()

        if len(down_features) != len(up_features):
            raise ValueError

        self.in_features = in_features
        self.out_features = out_features
        # avoid bottlenecks
        hidden_features = max(in_features, out_features)
        self.hidden_features = hidden_features

        self.encoder = ConvBlock(
            down_features[0],
            in_features=in_features,
            out_features=hidden_features,
        )
        self.decoder = ConvBlock(
            up_features[-1],
            in_features=hidden_features,
            out_features=out_features,
        )

        self.encoder_pooling = nn.Conv2d(
            hidden_features, hidden_features, 3, 2, padding=1,
        )
        self.decoder_pooling = nn.ConvTranspose2d(
            hidden_features, hidden_features, 3, 2, padding=1,
        )

        if in_features == out_features:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Conv2d(
                in_features,
                out_features,
                1,
                1,
                bias=False,
                padding="same",
            )
            nn.init.orthogonal_(self.projector.weight.data)

        if down_features[1:]:
            # create the next level
            self.unet = UNet(
                in_features=hidden_features,
                out_features=hidden_features,
                down_features=down_features[1:],
                up_features=up_features[:-1],
                activation=activation,
            )
        else:
            self.unet = nn.Identity()

    def __call__(self, x, **kwargs):
        # pad so we don't lose information on the down-pooling
        x = pad_to_even_height_and_width(x)

        residual = self.projector(x)

        x = self.encoder(x, **kwargs)
        x = self.encoder_pooling(x)
        x = self.unet(x, **kwargs)
        x = self.decoder_pooling(x)
        x = self.decoder(x, **kwargs)

        # center-crop after the forward pass to discard unusable pixels
        x = center_crop(x, residual.shape[-2:])

        return x + residual
