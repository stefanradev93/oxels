from collections.abc import Sequence
import os

from torch.utils.data import DataLoader
from torchvision import transforms

from oxels.datasets import CIFAR10 as ImageNet
from oxels.networks import SimpleUNet, SimpleUViT, UNet

from .base_model import BaseModel


class ImageNetModel(BaseModel):
    def __init__(
        self,
        *,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        num_res_blocks: Sequence[int] = (2, 2, 2, 4),
        num_oxels: int = 64,
        num_norm_groups: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.004,
        dropout_stages: Sequence[int],
        dropout: float = 0.1,
        attention_stages: Sequence[int],
        lr_div_factor: float = 25.0,
        lr_final_div_factor: float = 1e4,
        total_steps: int,
        train_batch_size: int,
        val_batch_size: int,
        image_size: int = 256,
    ):
        num_stages = len(stage_channels)
        has_attention = [False] * num_stages
        for stage in attention_stages:
            has_attention[stage] = True

        residual_dropout = [0.0] * num_stages
        for stage in dropout_stages:
            residual_dropout[stage] = dropout

        # backbone = SimpleUNet(
        #     height=image_size,
        #     width=image_size,
        #     in_channels=3,
        #     out_channels=num_oxels,
        #     channels_of_stage=stage_channels,
        #     has_attention=has_attention,
        #     num_res_blocks=num_res_blocks,
        #     norm_groups=num_norm_groups,
        #     residual_dropout=residual_dropout,
        # )

        down_features = [[c] * n for c, n in zip(stage_channels, num_res_blocks)]
        middle_features = [stage_channels[-1]] * 3
        up_features = list(reversed([list(reversed(c)) for c in down_features]))

        backbone = UNet(
            in_features=3,
            out_features=num_oxels,
            down_features=down_features,
            middle_features=middle_features,
            up_features=up_features,
        )

        # backbone = SimpleUViT(
        #     height=image_size,
        #     width=image_size,
        #     in_channels=3,
        #     out_channels=num_oxels,
        #     channels_of_stage=stage_channels,
        #     num_res_blocks=num_res_blocks,
        #     norm_groups=num_norm_groups,
        #     residual_dropout=residual_dropout,
        # )

        super().__init__(
            backbone=backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_div_factor=lr_div_factor,
            lr_final_div_factor=lr_final_div_factor,
            total_steps=total_steps,
        )

        self.save_hyperparameters()

    def train_dataloader(self):
        if self.hparams.image_size != 256:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.hparams.image_size)])
        else:
            transform = None

        dataset = ImageNet(split="train", transform=transform)

        return DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True,
        )

    def val_dataloader(self):
        if self.hparams.image_size != 256:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.hparams.image_size)])
        else:
            transform = None

        dataset = ImageNet(split="val", transform=transform)

        return DataLoader(
            dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=False,
        )
