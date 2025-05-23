import lightning as L
import torch.nn as nn
from torch import compile as jit

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from oxels.losses import vectorized_loss

from .metrics_mixin import MetricsMixin


class BaseModel(MetricsMixin, L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.004,
        lr_pct_start: float = 0.05,
        lr_div_factor: float = 25.0,
        lr_final_div_factor: float = 1e4,
    ):
        super().__init__()
        self.save_hyperparameters(
            learning_rate, weight_decay, lr_pct_start, lr_div_factor, lr_final_div_factor, ignore=["backbone"]
        )
        self.backbone = backbone

    @jit
    def forward(self, x):
        return self.backbone(x)

    @jit
    def compute_loss(self, batch):
        view1, view2, permutation, flags, mask1, mask2 = batch

        oxels_view1 = self(view1)
        oxels_view2 = self(view2)

        loss = vectorized_loss(oxels_view1, oxels_view2, permutation, flags, mask1, mask2)

        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay

        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.99))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.hparams.lr_div_factor,
            final_div_factor=self.hparams.lr_final_div_factor,
            pct_start=self.hparams.lr_pct_start,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
