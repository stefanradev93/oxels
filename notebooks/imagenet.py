import lightning as L
import optuna
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger

import wandb
from oxels.callbacks import ReportValidationLoss, ShowOxels
from oxels.models import ImageNetModel


def objective(trial: optuna.Trial):
    torch.set_float32_matmul_precision("medium")

    total_steps = 300_000
    trial_steps = 10_000
    image_size = 64
    num_nodes = 1
    train_batch_size = 12
    val_batch_size = 24
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    weight_decay = 1e-4

    num_stages = trial.suggest_int("num_stages", 4, 4)
    num_oxels = trial.suggest_int("num_oxels", 16, 64, step=8)
    base_channels = trial.suggest_int("base_channels", max(16, num_oxels), 64, step=8)

    stage_multipliers = [
        1,
        trial.suggest_int("stage_multiplier_1", 1, 2),
        trial.suggest_int("stage_multiplier_2", 2, 4, step=2),
        trial.suggest_int("stage_multiplier_3", 2, 4, step=2),
        trial.suggest_int("stage_multiplier_4", 4, 8, step=2),
        trial.suggest_int("stage_multiplier_5", 4, 8, step=2),
    ]
    stage_multipliers = stage_multipliers[:num_stages]
    stage_channels = [base_channels * factor for factor in stage_multipliers]

    num_res_blocks = [
        1,
        trial.suggest_int("num_res_blocks_1", 1, 2, step=1),
        trial.suggest_int("num_res_blocks_2", 2, 4, step=2),
        trial.suggest_int("num_res_blocks_3", 2, 6, step=2),
        trial.suggest_int("num_res_blocks_4", 2, 8, step=2),
        trial.suggest_int("num_res_blocks_5", 2, 8, step=2),
    ]

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
        max_steps=trial_steps,
        accelerator="gpu",
        strategy="ddp",
        devices=-1,
        precision="16-mixed",
        num_nodes=num_nodes,
    )

    model = ImageNetModel(**model_config)

    with torch.device("cpu"):
        validation_images = [model.val_dataloader().dataset.dataset[i][0] for i in range(4)]
        validation_images = torch.stack(validation_images)

    num_parameters = sum(p.numel() for p in model.parameters())

    config = model_config | trainer_config

    run = wandb.init(
        entity="kl_divergence-rensselaer-polytechnic-institute",
        project="oxels",
        name="ImageNet Hyperparameter Tuning",
        config=config,
        dir="wandb_results",
    )

    wandb.summary["trial_steps"] = trial_steps
    wandb.summary["num_parameters"] = num_parameters

    logger = WandbLogger(
        name="ImageNet Hyperparameter Tuning",
        save_dir="logs",
        project="oxels",
    )

    trainer = L.Trainer(
        **trainer_config,
        callbacks=[
            callbacks.LearningRateMonitor(logging_interval="step"),
            #            callbacks.ModelCheckpoint(
            #                monitor="validation/loss",
            #                mode="min",
            #                save_top_k=1,
            #                filename="best_model",
            #            ),
            ReportValidationLoss(trial),
            ShowOxels(images=validation_images),
        ],
        logger=logger,
    )

    trainer.fit(model)

    metrics = trainer.validate(model)[0]

    wandb.summary["validation/loss"] = metrics["validation/loss"]

    run.finish()

    return metrics["validation/loss"]


pruner = optuna.pruners.HyperbandPruner()
pruner = optuna.pruners.PatientPruner(pruner, patience=128, min_delta=1e-3)

study = optuna.create_study(direction="minimize", pruner=pruner, storage="sqlite:///imagenet.db", load_if_exists=True)
study.optimize(objective, n_trials=30, catch=RuntimeError)
