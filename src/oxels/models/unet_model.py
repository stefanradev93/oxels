from collections.abc import Sequence
from typing import Optional

import torch.nn as nn

from oxels.networks import UNet
from oxels.networks.helpers import ConvBlock
from oxels.typing import ActivationType, UNetBlockType as BlockType

from .base_model import BaseModel


class UNetModel(BaseModel):
    default_down_features = ((8, 16), (24, 32), (40, 48), (56, 64))
    default_middle_features = (64, 64, 64)
    default_up_features = ((64, 56), (48, 40), (32, 24), (16, 8))

    def __init__(
        self,
        *,
        in_features: int = 3,
        out_features: int = 8,
        down_features: Sequence[Sequence[int]] = default_down_features,
        middle_features: Sequence[int] = default_middle_features,
        up_features: Sequence[Sequence[int]] = default_up_features,
        activation: ActivationType = nn.SiLU,
        out_activation: Optional[ActivationType] = None,
        down_types: BlockType | Sequence[BlockType] = ConvBlock,
        middle_type: BlockType = ConvBlock,
        up_types: BlockType | Sequence[BlockType] = ConvBlock,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.004,
        lr_div_factor: float = 25.0,
        lr_final_div_factor: float = 1e4,
        lr_pct_start: float = 0.05,
    ):
        backbone = UNet(
            in_features=in_features,
            out_features=out_features,
            down_features=down_features,
            middle_features=middle_features,
            up_features=up_features,
            activation=activation,
            out_activation=out_activation,
            down_types=down_types,
            middle_type=middle_type,
            up_types=up_types,
        )
        super().__init__(
            backbone=backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_div_factor=lr_div_factor,
            lr_final_div_factor=lr_final_div_factor,
            lr_pct_start=lr_pct_start,
        )
        self.save_hyperparameters()
