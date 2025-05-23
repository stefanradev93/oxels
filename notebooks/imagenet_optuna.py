
import lightning as L
import optuna
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.loggers import WandbLogger

import os
import sys
import wandb

from functools import partial


from oxels.callbacks import ShowOxels
from oxels.distribution import rank_zero, send_or_recv
from oxels.models import ImageNetModel

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


def sample_configs(trial: optuna.Trial):
    total_steps = 10_000
    warmup_steps = 1_000
    train_batch_size = 24
    val_batch_size = 64
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    lr_pct_start = warmup_steps / total_steps
    weight_decay = 1e-4

    num_stages = trial.suggest_int("num_stages", 4, 6)
    num_oxels = trial.suggest_int("num_oxels", 32, 96, step=8)
    # base_channels = trial.suggest_int("base_channels", 32, 96, step=16)
    stage_channels = [
        trial.suggest_int("stage_channels_0", num_oxels, max(num_oxels, 96), step=32),
        trial.suggest_int("stage_channels_1", max(num_oxels, 64), 128, step=32),
        trial.suggest_int("stage_channels_2", 96, 192, step=32),
        trial.suggest_int("stage_channels_3", 128, 256, step=32),
        trial.suggest_int("stage_channels_4", 160, 320, step=32),
        trial.suggest_int("stage_channels_5", 192, 384, step=32),
    ]
    num_res_blocks = [
        trial.suggest_int("num_res_blocks_0", 2, 4),
        trial.suggest_int("num_res_blocks_1", 2, 4),
        trial.suggest_int("num_res_blocks_2", 3, 6),
        trial.suggest_int("num_res_blocks_3", 4, 8),
        trial.suggest_int("num_res_blocks_4", 6, 10),
        trial.suggest_int("num_res_blocks_5", 8, 12),
    ]

    stage_channels = stage_channels[:num_stages]
    num_res_blocks = num_res_blocks[:num_stages]

    dropout_stages = [-2, -1]
    dropout = 0.1

    attention_stages = [-2, -1]

    model_config = dict(
        stage_channels=stage_channels,
        num_res_blocks=num_res_blocks,
        num_oxels=num_oxels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout_stages=dropout_stages,
        dropout=dropout,
        attention_stages=attention_stages,
        lr_div_factor=25.0,
        lr_final_div_factor=1e4,
        lr_pct_start=lr_pct_start,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        image_size=256,
    )

    trainer_config = dict(
        max_steps=total_steps,
        max_time="00:01:29:00",
        accelerator="gpu",
        num_nodes=count_nodes(),
        devices=-1,
        strategy="ddp",
        gradient_clip_val=3.0,
        gradient_clip_algorithm="value",
        precision="16-mixed",
        enable_checkpointing=False,
    )

    return model_config, trainer_config


def objective(trial: optuna.Trial):
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    model_config, trainer_config = send_or_recv(partial(sample_configs, trial))

    model = ImageNetModel(**model_config)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    callbacks = [
        LearningRateMonitor(),
        PyTorchLightningPruningCallback(trial, monitor="validation/loss"),
    ]

    if dist.get_rank() == 0:
        with torch.device("cpu"):
            validation_images = [model.val_dataloader().dataset.dataset[i][0] for i in range(4)]
            validation_images = torch.stack(validation_images)

        show_oxels = ShowOxels(validation_images)
        callbacks.append(show_oxels)

    run = wandb.init(
        entity="kl_divergence-rensselaer-polytechnic-institute",
        project="oxels",
        config=model_config | trainer_config,
        dir="wandb_results",
    )

    logger = WandbLogger(experiment=run)

    wandb.summary["num_parameters"] = num_parameters

    trainer = L.Trainer(**trainer_config, callbacks=callbacks, logger=logger)

    try:
        trainer.fit(model)
        return trainer.callback_metrics["validation/loss"]
    finally:
        run.finish()


@rank_zero(error=True)
def create_study():
    pruner = optuna.pruners.HyperbandPruner()
    pruner = optuna.pruners.PatientPruner(pruner, patience=3, min_delta=1e-3)

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///imagenet-optuna.db",
        study_name="imagenet-optuna",
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

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

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

        # TODO: make this call less ambiguous
        #  this essentially only updates the study on rank zero
        _ = send_or_recv(lambda: study.tell(trial, value))

        # maybe superfluous:
        dist.barrier()

    return 0


if __name__ == "__main__":
    exit(main(sys.argv))
