import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

import torch
import wandb

from pathlib import Path

from oxels.callbacks import ShowOxels
from oxels.distribution import count_nodes, get_rank, get_world_size
from oxels.models import ImageNetModel


def setup():
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")


def get_configs():
    max_epochs = None
    max_steps = 62_500
    warmup_steps = 1_250
    train_batch_size = 16
    val_batch_size = 32

    target_batch_size = 2048
    accumulate_grad_batches = int(target_batch_size / (train_batch_size * get_world_size()))
    accumulate_grad_batches = max(1, accumulate_grad_batches)
    learning_rate = 1e-3
    lr_pct_start = warmup_steps / max_steps
    weight_decay = 1e-4

    num_oxels = 64

    stage_channels = [64, 96, 128, 192, 256]
    num_res_blocks = [2, 2, 4, 6, 6]

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
        frac_keep=0.25,
        contrastive_loss_weight=0.75,
    )

    trainer_config = dict(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="gpu",
        num_nodes=count_nodes(),
        devices=-1,
        strategy="ddp",
        gradient_clip_val=3.0,
        gradient_clip_algorithm="value",
        precision="16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=0.1,
    )

    return model_config, trainer_config


def train(model_config, trainer_config):
    model = ImageNetModel(**model_config)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dirpath = "checkpoints"
    filename = "imagenet.ckpt"
    ckpt_path = Path(dirpath) / filename
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=str(ckpt_path.parent),
            filename=ckpt_path.stem,
            monitor="validation/loss",
            save_top_k=5,
            save_last=True,
            save_on_train_epoch_end=True,
            mode="min",
        ),
    ]

    last_ckpt_path = ckpt_path.with_stem("last")
    if last_ckpt_path.is_file():
        print(f"Loading last checkpoint from {last_ckpt_path}")
        ckpt_path = last_ckpt_path
    elif not ckpt_path.is_file():
        print(f"Checkpoint {ckpt_path} not found, starting from scratch.")
        ckpt_path = None

    if get_rank() == 0:
        with torch.device("cpu"):
            validation_images = [model.val_dataloader().dataset.dataset[i][0] for i in range(4)]
            validation_images = torch.stack(validation_images)

        show_oxels = ShowOxels(validation_images)
        callbacks.append(show_oxels)

        logger = WandbLogger(
            name="imagenet-contrastive",
            project="oxels",
            save_dir="logs",
            id="imagenet-contrastive",
            entity="kl_divergence-rensselaer-polytechnic-institute",
            config=model_config | trainer_config,
            settings=wandb.Settings(init_timeout=1800),
            mode="online",
            resume="allow",
        )

        # logger.experiment.summary["num_parameters"] = num_parameters
        # logger.experiment.summary["effective_batch_size"] = model_config["train_batch_size"] * trainer_config["accumulate_grad_batches"] * get_world_size()
    else:
        logger = WandbLogger(
            name="imagenet-contrastive",
            project="oxels",
            save_dir="logs",
            id="imagenet-contrastive",
            entity="kl_divergence-rensselaer-polytechnic-institute",
            config=model_config | trainer_config,
            settings=wandb.Settings(init_timeout=1800),
            mode="disabled",
            resume="allow",
        )

    trainer = L.Trainer(**trainer_config, callbacks=callbacks, logger=logger)

    trainer.fit(model, ckpt_path=ckpt_path)


def main(args):
    setup()

    model_config, trainer_config = get_configs()
    train(model_config, trainer_config)

    return 0


if __name__ == "__main__":
    import sys

    exit(main(sys.argv))
