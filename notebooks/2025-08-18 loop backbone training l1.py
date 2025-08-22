from oxels.datasets import DGPerspectiveDataset
import matplotlib.pyplot as plt
from collections.abc import Sequence
import os
from torch.utils.data import DataLoader
from torchvision import transforms

from oxels.networks import SimpleUNet
from oxels.models.base_model import BaseModel
from oxels.datasets import DGPerspectiveDataset
import lightning as L
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger

import wandb
from oxels.callbacks import ShowOxels

import numpy as np
norms = ["l1"]
frac_keep = 0.25
contrastive_loss_weights = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
num_oxels = [32, 32, 32]

for i, loss_norm_type in enumerate(norms):
    for j, contrastive_loss_weight in enumerate(contrastive_loss_weights):
        seed = i * len(num_oxels) + j
        dataset = DGPerspectiveDataset(
            path="datasets",
            dataset="ColoredMNIST",
            h=32,
            w=32,
            split="test",
            domain_split="id",
            seed=seed,
        )

        class ColoredMNISTModel(BaseModel):
            def __init__(
                self,
                *,
                stage_channels: Sequence[int] = (32, 64, 128, 256),
                num_res_blocks: Sequence[int] = (2, 2, 2, 4),
                num_oxels: int = 64,
                num_norm_groups: int = 8,
                learning_rate: float = 1e-3,
                weight_decay: float = 0.004,
                dropout_stages: Sequence[int],
                dropout: float = 0.1,
                attention_stages: Sequence[int],
                lr_div_factor: float = 25.0,
                lr_final_div_factor: float = 1e4,
                lr_pct_start: float = 0.05,
                train_batch_size: int,
                val_batch_size: int,
                image_size: int = 256,
                contrastive_loss_weight: float = 0.5,
                frac_keep: float = 0.125,
                loss_norm_type: str = "inf",
            ):
                num_stages = len(stage_channels)
                has_attention = [False] * num_stages
                for stage in attention_stages:
                    has_attention[stage] = True

                residual_dropout = [0.0] * num_stages
                for stage in dropout_stages:
                    residual_dropout[stage] = dropout

                backbone = SimpleUNet(
                    height=image_size,
                    width=image_size,
                    in_channels=3,
                    out_channels=num_oxels,
                    channels_of_stage=stage_channels,
                    has_attention=has_attention,
                    num_res_blocks=num_res_blocks,
                    norm_groups=num_norm_groups,
                    residual_dropout=residual_dropout,
                )
                super().__init__(
                    backbone=backbone,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    lr_div_factor=lr_div_factor,
                    lr_final_div_factor=lr_final_div_factor,
                    lr_pct_start=lr_pct_start,
                    contrastive_loss_weight=contrastive_loss_weight,
                    loss_norm_type=loss_norm_type,
                )
                self.frac_keep = frac_keep
                self.save_hyperparameters()

            def train_dataloader(self):
                dataset = DGPerspectiveDataset(
                    path="datasets",
                    dataset="ColoredMNIST",
                    h=self.hparams.image_size,
                    w=self.hparams.image_size,
                    split="train",
                    domain_split="id",
                    seed=seed,
                    augmentations=transforms.Compose([
                        transforms.Resize((32, 32))
                    ]),
                    frac_keep=self.frac_keep,
                )
                return DataLoader(
                    dataset,
                    batch_size=self.hparams.train_batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=len(os.sched_getaffinity(0)),
                    drop_last=True,
                )

            def val_dataloader(self):
                dataset = DGPerspectiveDataset(
                    path="datasets",
                    dataset="ColoredMNIST",
                    h=self.hparams.image_size,
                    w=self.hparams.image_size,
                    split="val",
                    domain_split="id",
                    seed=0,
                    augmentations=transforms.Compose([
                        transforms.Resize((32, 32))
                    ]),
                    frac_keep=self.frac_keep,
                )
                return DataLoader(
                    dataset,
                    batch_size=self.hparams.val_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=len(os.sched_getaffinity(0)),
                    drop_last=False,
                )

            def val_dataloader_ood(self):
                dataset = DGPerspectiveDataset(
                    path="datasets",
                    dataset="ColoredMNIST",
                    h=self.hparams.image_size,
                    w=self.hparams.image_size,
                    split="val",
                    domain_split="ood",
                    seed=0,
                    augmentations=transforms.Compose([
                        transforms.Resize((32, 32))
                    ]),
                    frac_keep=self.frac_keep,
                )
                return DataLoader(
                    dataset,
                    batch_size=self.hparams.val_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=len(os.sched_getaffinity(0)),
                    drop_last=False,
                )
        # ---- Training Configuration ----
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")

        print("Device Count:", torch.cuda.device_count())
        project = "ColoredMNIST-testing"
        total_steps = 30_000
        image_size = 32
        num_nodes = 1
        num_devices = 1
        train_batch_size = 128
        val_batch_size = 256
        learning_rate = 1e-3
        lr_pct_start = 0.05
        weight_decay = 1e-6

        num_stages = 3
        base_channels = 32
        stage_channels = [32, 64, 64, 128]
        num_res_blocks = [1, 2, 2, 4]
        stage_channels = stage_channels[:num_stages]
        num_res_blocks = num_res_blocks[:num_stages]

        dropout_stages = [-2, -1]
        dropout = 0.1
        attention_stages = [stage for stage in range(4, num_stages)]

        num_oxel = 32

        model_config = dict(
            stage_channels=stage_channels,
            num_res_blocks=num_res_blocks,
            num_oxels=num_oxel,
            dropout_stages=dropout_stages,
            dropout=dropout,
            attention_stages=attention_stages,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_pct_start=lr_pct_start,
            lr_div_factor=25.0,
            lr_final_div_factor=1e4,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            image_size=image_size,
            contrastive_loss_weight=contrastive_loss_weight,
            frac_keep=frac_keep,
            loss_norm_type=loss_norm_type,
        )

        trainer_config = dict(
            gradient_clip_val=3.0,
            gradient_clip_algorithm="value",
            max_steps=total_steps,
            accelerator="gpu",
            strategy="auto",
            devices=num_devices,
            precision="16-mixed",
            num_nodes=num_nodes,
        )

        model = ColoredMNISTModel(**model_config)

        with torch.device("cpu"):
            train_images = [model.train_dataloader().dataset.dataset[i][0] for i in range(4)]
            train_images = torch.stack(train_images)
            validation_images = [model.val_dataloader().dataset.dataset[i][0] for i in range(4)]
            validation_images = torch.stack(validation_images)
            validation_images_ood = [model.val_dataloader_ood().dataset.dataset[i][0] for i in range(4)]
            validation_images_ood = torch.stack(validation_images_ood)

        num_parameters = sum(p.numel() for p in model.parameters())

        config = model_config | trainer_config

        # ---- Initialize WandB run ----
        run = wandb.init(
            entity="kl_divergence-rensselaer-polytechnic-institute",
            project=project,
            config=config,
            dir="wandb_results",
        )

        logger = WandbLogger(
            experiment=run,
        )

        wandb.summary["num_parameters"] = num_parameters
        wandb.summary["model_type"] = "BackboneColoredMNIST"
        run_name = wandb.run.name
        run_id = wandb.run.id
        print(f"name: {run_name} \t run id:{run_id}")

        # ---- Training ----
        ckpt_dir = os.path.join("lightning_logs", str(run_name))
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "wandb_run_id.txt"), "w") as f:
            f.write(run_id)
        f.close()
        trainer = L.Trainer(
            **trainer_config,
            callbacks=[
                callbacks.LearningRateMonitor(logging_interval="step"),
                callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir,
                    monitor="validation/loss",
                    mode="min",
                    save_top_k=1,
                    filename="best_model",
                    save_last=True,
                ),
                ShowOxels(images=train_images, every_n_epochs=10, caption="Train Oxels"),
                ShowOxels(images=validation_images, every_n_epochs=10, caption="Validation Oxels"),
                ShowOxels(images=validation_images_ood, every_n_epochs=10, caption="Validation OOD Oxels"),
            ],
            logger=logger,
        )

        try:
            trainer.fit(model)

            result = trainer.callback_metrics["validation/loss"]
            wandb.summary["result"] = result
            del trainer, model, logger, run, dataset
            torch.cuda.empty_cache()

        finally:
            # clean up
            wandb.finish()

