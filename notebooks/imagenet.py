import lightning as L
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger

import wandb
from oxels.callbacks import ShowOxels
from oxels.models import ImageNetModel


def objective():
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    print("Device Count:", torch.cuda.device_count())

    total_steps = 300_000
    image_size = 64
    num_nodes = 1
    num_devices = 4
    train_batch_size = 128
    val_batch_size = 256
    learning_rate = 3e-4
    lr_pct_start = 0.05
    weight_decay = 1e-4

    num_stages = 4
    num_oxels = 64
    base_channels = 64
    stage_channels = [64, 128, 128, 256]
    num_res_blocks = [1, 2, 4, 6]

    dropout_stages = [-2, -1]
    dropout = 0.1

    attention_stages = [stage for stage in range(4, num_stages)]

    model_config = dict(
        stage_channels=stage_channels,
        num_res_blocks=num_res_blocks,
        num_oxels=num_oxels,
        dropout_stages=dropout_stages,
        dropout=dropout,
        attention_stages=attention_stages,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_pct_start=lr_pct_start,
        lr_div_factor=25.0,
        lr_final_div_factor=1e4,
        total_steps=total_steps,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        image_size=image_size,
    )

    trainer_config = dict(
        gradient_clip_val=3.0,
        gradient_clip_algorithm="value",
        max_steps=total_steps,
        accelerator="gpu",
        strategy="ddp",
        devices=num_devices,
        precision="16-mixed",
        num_nodes=num_nodes,
    )

    model = ImageNetModel(**model_config)

    with torch.device("cpu"):
        validation_images = [model.val_dataloader().dataset.dataset[i][0] for i in range(4)]
        validation_images = torch.stack(validation_images)

    num_parameters = sum(p.numel() for p in model.parameters())

    config = model_config | trainer_config

    # grab the few key trial.params you care about
    lr = learning_rate
    bc = base_channels
    ns = num_stages
    ox = num_oxels
    n_params = num_parameters / 1e6

    # build a short, human-readable name
    run_name = (
        f"CIFAR64-lr{lr:.0e}_bc{bc}_stg{ns}_ox{ox}_"
        f"{n_params:.1f}M"
    )

    run = wandb.init(
        entity="kl_divergence-rensselaer-polytechnic-institute",
        project="oxels",
        config=config,
        name=run_name,
        dir="wandb_results",
    )

    wandb.define_metric("training/step")
    wandb.define_metric("validation/step")
    wandb.define_metric("testing/step")

    wandb.define_metric("training/*", step_metric="training/step")
    wandb.define_metric("validation/*", step_metric="validation/step")
    wandb.define_metric("testing/*", step_metric="testing/step")

    wandb.summary["num_parameters"] = num_parameters

    logger = WandbLogger(
        save_dir="logs",
        project="oxels",
        name=run_name,
    )

    trainer = L.Trainer(
        **trainer_config,
        callbacks=[
            callbacks.LearningRateMonitor(logging_interval="step"),
            callbacks.ModelCheckpoint(
                monitor="validation/loss",
                mode="min",
                save_top_k=1,
                filename="best_model",
            ),
            ShowOxels(images=validation_images),
        ],
        logger=logger,
        # val_check_interval=0.1,
    )

    try:
        trainer.fit(model)

        result = trainer.callback_metrics["validation/loss"]
        wandb.summary["result"] = result

        return result
    finally:
        # clean up
        run.finish()


def main(args):
    objective()


if __name__ == "__main__":
    import sys

    exit(main(sys.argv))
