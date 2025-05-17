from collections.abc import Sequence

import torch
import torch.nn as nn

from typing import Optional

from oxels.typing import ActivationType


class AttentionLayer(nn.Module):
    def __init__(self, *, in_features: int, out_features: int, num_heads: int = 4, activation: ActivationType = nn.SiLU):
        super().__init__()
        hidden_features = num_heads * max(in_features, out_features)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_heads = num_heads

        self.residual_projector = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding="same", bias=False)
        nn.init.orthogonal_(self.residual_projector.weight)

        self.input_projector = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding="same")
        nn.init.orthogonal_(self.input_projector.weight)
        nn.init.zeros_(self.input_projector.bias)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_features, num_heads=num_heads)
        # nn.init.orthogonal_(self.attention.in_proj_weight)
        # nn.init.zeros_(self.attention.in_proj_bias)
        # nn.init.orthogonal_(self.attention.out_proj.weight)
        # nn.init.zeros_(self.attention.out_proj.bias)

        self.layer_norm = nn.LayerNorm(hidden_features)
        self.output_projector = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding="same")
        nn.init.orthogonal_(self.output_projector.weight)
        nn.init.zeros_(self.output_projector.bias)

        if activation is not None:
            self.activation = activation()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        residual = self.residual_projector(x)

        x = self.input_projector(x)

        # (B, C, H, W) -> (B, H * W, C)
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        x = self.attention(x, x, x)[0]
        x = self.layer_norm(x)

        # (B, H * W, C) -> (B, C, H, W)
        x = x.permute(0, 2, 1).view(batch_size, channels, height, width)

        # ensure contiguous conv input to avoid performance issues with ddp
        x = x.contiguous()

        x = self.output_projector(x)
        x = self.activation(x)

        x = x + residual

        return x

        # ensure we return contiguous output


class AttentionBlock(nn.Sequential):
    def __init__(
        self,
        hidden_features: Sequence[int],
        *,
        in_features: int,
        out_features: int,
        activation: ActivationType = nn.SiLU,
        out_activation: Optional[ActivationType] = None,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        input_layer = AttentionLayer(
            in_features=in_features,
            out_features=hidden_features[0],
            num_heads=num_heads,
            activation=activation,
        )
        self.append(input_layer)

        for in_features, out_features in zip(hidden_features[:-1], hidden_features[1:]):
            hidden_layer = AttentionLayer(
                in_features=in_features,
                out_features=out_features,
                num_heads=num_heads,
                activation=activation,
            )
            self.append(hidden_layer)

        output_layer = AttentionLayer(
            in_features=hidden_features[-1],
            out_features=out_features,
            num_heads=num_heads,
            activation=out_activation,
        )
        self.append(output_layer)
