from typing import Optional, Union

import lightning as L
import torch
import torch.nn as nn
from torch import compile as jit

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from oxels.losses import original_loss

from .metrics_mixin import MetricsMixin


class BaseModel(MetricsMixin, L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.004,
        lr_div_factor: float = 25.0,
        lr_final_div_factor: float = 1e4,
        total_steps: int,
    ):
        super().__init__()
        self.save_hyperparameters(learning_rate, weight_decay, lr_div_factor, lr_final_div_factor, total_steps, ignore=["backbone"])
        self.backbone = backbone

    @jit
    def compute_loss(self, batch):
        view1, view2, permutation, flags, mask1, mask2 = batch

        oxels_view1 = self.backbone(view1)
        oxels_view2 = self.backbone(view2)

        loss = original_loss(oxels_view1, oxels_view2, permutation, flags, mask1, mask2)

        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay

        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.hparams.total_steps,
            div_factor=self.hparams.lr_div_factor,
            final_div_factor=self.hparams.lr_final_div_factor,
        )

        return [optimizer], [scheduler]
