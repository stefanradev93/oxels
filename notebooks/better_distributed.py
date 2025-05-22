
import lightning as L
import optuna
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import os
import sys


from oxels.distribution import rank_zero, send_or_recv


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


def count_nodes() -> int:
    return int(os.environ.get("SLURM_NNODES", 1))


def get_local_rank() -> int:
    return int(os.environ.get("SLURM_PROCID", 0)) % int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))


def get_rank() -> int:
    return int(os.environ.get("SLURM_PROCID", 0))


def get_world_size() -> int:
    return int(os.environ.get("SLURM_NTASKS", 1))


def get_job_id() -> int:
    return int(os.environ.get("SLURM_JOB_ID", 0))


def objective(trial: optuna.Trial):
    torch.cuda.empty_cache()

    dim = trial.suggest_int("dim", 1, 128)

    dim = send_or_recv(lambda: dim)

    model = MinimalModel(dim)
    trainer = L.Trainer(
        max_steps=100,
        accelerator="gpu",
        num_nodes=count_nodes(),
        devices=-1,
        strategy="ddp",
        gradient_clip_val=3.0,
        gradient_clip_algorithm="value",
        precision="16-mixed",
    )

    trainer.fit(model)

    return trainer.callback_metrics["validation/loss"]


@rank_zero(error=True)
def create_study():
    pruner = optuna.pruners.HyperbandPruner()
    pruner = optuna.pruners.PatientPruner(pruner, patience=3, min_delta=1e-3)

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///study.db",
        study_name="study",
        load_if_exists=True,
    )

    return study


def setup():
    # set the seed for reproducibility
    seed = get_job_id()
    L.seed_everything(seed)

    # get environment info from slurm or fallback to a single device
    rank = get_rank()
    size = get_world_size()

    # set torch environment variables
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)

    # initialize the process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)

    # set the device for this process
    local_rank = get_local_rank()
    torch.cuda.set_device(f"cuda:{local_rank}")


def main(args):
    setup()

    size = dist.get_world_size()

    study = send_or_recv(create_study)

    while True:
        trial = send_or_recv(study.ask)
        value = objective(trial)

        # aggregate the value across all ranks
        value = torch.tensor(value)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = float(value) / size

        study.tell(trial, value)

    return 0


if __name__ == "__main__":
    exit(main(sys.argv))
