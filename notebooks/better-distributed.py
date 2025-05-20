
import lightning as L
import optuna
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import os
import sys

from .distutils import rank_zero


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


@rank_zero(error=True)
def broadcast_object(obj: any) -> None:
    if dist.get_world_size() > 1:
        object_list = [obj]
        for _rank in range(1, dist.get_world_size()):
            dist.send_object_list(object_list, dst=_rank)


def recv_object() -> any:
    rank = dist.get_rank()
    if rank == 0:
        raise RuntimeError(f"This function should never be called by rank 0")

    object_list = [None]
    dist.recv_object_list(object_list, src=0)

    return object_list[0]


def create_or_recv_study(rank=None, group=None) -> optuna.Study:
    if rank is None:
        rank = dist.get_rank(group)

    if rank == 0:
        study = create_study()
        broadcast_object(study)
    else:
        study = recv_object()

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

    study = create_or_recv_study()
    study.optimize(objective, gc_after_trial=True)

    return 0


if __name__ == "__main__":
    exit(main(sys.argv))
