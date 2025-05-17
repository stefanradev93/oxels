
import lightning as L
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class MinimalModel(L.LightningModule):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.linear(x)
        loss = F.mse_loss(yhat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        x = torch.randn((1024, self.dim))
        y = 2 * x + 3
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=32, shuffle=True)


def objective(trial: optuna.Trial):
    dim = trial.suggest_int("dim", 3, 5)
    model = MinimalModel(dim)
    trainer = L.Trainer(
        max_steps=100,
        accelerator="gpu",
        num_modes=2,
        devices=-1,
        strategy="ddp",
        precision="16-mixed",
    )
    trainer.fit(model)


def main(args):
    import os

    seed = int(os.environ.get("SLURM_JOB_ID", 0))
    L.seed_everything(seed)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name="minimal-model",
        storage="sqlite:///minimal-model.db",
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(objective)


if __name__ == "__main__":
    import sys

    main(sys.argv)
