import lightning as L
import torch.nn as nn

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
        self.save_hyperparameters(
            learning_rate, weight_decay, lr_div_factor, lr_final_div_factor, total_steps, ignore=["backbone"]
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def compute_loss(self, batch):
        view1, view2, permutation, flags, mask1, mask2 = batch

        oxels_view1 = self(view1)
        oxels_view2 = self(view2)

        loss = original_loss(oxels_view1, oxels_view2, permutation, flags, mask1, mask2)

        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay

        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.99))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.hparams.total_steps,
            div_factor=self.hparams.lr_div_factor,
            final_div_factor=self.hparams.lr_final_div_factor,
        )

        return [optimizer], [scheduler]
