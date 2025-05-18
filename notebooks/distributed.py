
import optuna
import os
import torch
import torch.distributed as dist

import lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MinimalModel(L.LightningModule):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.linear(x)
        loss = F.mse_loss(yhat, y)
        self.log("validation/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        x = torch.randn((1024, self.dim))
        y = 2 * x + 3
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=32, shuffle=True)


def connect(model_config, trainer_config):
    print(f"Connecting process {dist.get_rank()} to the trainer...")
    model = MinimalModel(**model_config)

    trainer = L.Trainer(**trainer_config)

    trainer.fit(model)

    return trainer.callback_metrics["validation/loss"]


def objective(trial):
    if dist.get_rank() != 0:
        raise RuntimeError("This function should only be called by rank 0")

    print(f"Initializing hyperparameters for trial {trial.number} on rank {dist.get_rank()}...")

    model_config = dict(
        dim=trial.suggest_int("dim", 3, 5)
    )
    trainer_config = dict(
        max_steps=100,
        accelerator="gpu",
        devices=-1,
        num_nodes=2,
        strategy="ddp",
    )

    if dist.get_world_size() > 1:
        print(f"Broadcasting hyperparameters for trial {trial.number} across all ranks...")
        object_list = [model_config, trainer_config]
        # broadcast the hyperparameters to all ranks
        for _rank in range(1, dist.get_world_size()):
            dist.send_object_list(object_list, dst=_rank)
        print(f"Successfully broadcasted hyperparameters for trial {trial.number}!")

    return connect(model_config, trainer_config)


def run(rank: int, size: int):
    print(f"Called run on rank {rank}.")
    if rank == 0:
        # only rank 0 creates the study and optimizes it
        print(f"Creating storage and study on rank {rank}...")
        storage = "sqlite:///storage.db"
        study = optuna.create_study(direction="minimize", storage=storage)

        print(f"Starting optimization on rank {rank}...")
        study.optimize(objective)
        return 0

    # other ranks keep connecting to the study
    while True:
        print(f"Waiting for hyperparameters on rank {rank}...")
        # other ranks only connect to the study
        object_list = []
        dist.recv_object_list(object_list, src=0)
        model_config, trainer_config = object_list
        print(f"Received hyperparameters on rank {rank}!")

        connect(model_config, trainer_config)


def main(args):
    # get environment info from slurm or fallback to a single device
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    size = int(os.environ.get("SLURM_NTASKS", "1"))

    print(f"Entering process with rank {rank} and size {size}...")

    # set torch environment variables
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)

    # define which node is the main node
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split(",")[0]

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    print(f"Master address: {os.environ['MASTER_ADDR']}")
    print(f"Master port: {os.environ['MASTER_PORT']}")

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=size)

    return run(rank, size)


if __name__ == "__main__":
    import sys

    exit(main(sys.argv))
