import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# Set memory fraction to avoid using all GPU memory
torch.cuda.set_per_process_memory_fraction(0.9)
# Empty cache before starting
torch.cuda.empty_cache()

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything

import wandb
from pathlib import Path

from oxels.callbacks import ShowOxels
from oxels.distribution import count_nodes, get_rank, get_world_size
from oxels.datasets.synthetic import Contrastive3D
from oxels.models.synthetic_model import Contrastive_Model

# Data root directory
DATA_ROOT = "./contrastive_3d_final"

def get_config():
    max_epochs = 20
    max_steps = 100_000
    warmup_steps = 2_000
    train_batch_size = 2
    val_batch_size = 2

    target_batch_size = 32
    accumulate_grad_batches = int(target_batch_size / (train_batch_size * get_world_size()))
    accumulate_grad_batches = max(1, accumulate_grad_batches)
    learning_rate = 1e-3
    lr_pct_start = .05
    weight_decay = 0.0

    num_oxels = 32

    stage_channels = [16, 16, 32, 32]	#[64, 48, 48, 32, 32]
    num_res_blocks = [1, 1, 1, 1]#[1, 1, 2, 2, 4]
    dropout_stages = [-2, -1]
    dropout = 0.1

    attention_stages = []

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
        contrastive_loss_weight=0.5,
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
        val_check_interval=0.05
    )

    return model_config, trainer_config


def setup():
    torch.set_float32_matmul_precision("medium")


def train(model_config, trainer_config):
    model = Contrastive_Model(data_root=DATA_ROOT, **model_config)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dirpath = "checkpoints"
    filename = "synthetic_fixed.ckpt"
    ckpt_path = Path(dirpath) / filename
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=str(ckpt_path.parent),
            filename=ckpt_path.stem,
            monitor="training/loss",
            save_top_k=5,
            save_last=True,
            save_on_train_epoch_end=True,
            mode="min",
        )
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
            # Get validation images directly from the dataset
            val_dataset = Contrastive3D(root=DATA_ROOT, split="val")
            validation_images = []
            for i in range(4):
                v1, v2, _, _, _, _ = val_dataset[i]
                validation_images.append(v1)  # Use first view
            validation_images = torch.stack(validation_images)
            
            # Debug: Check image values
            print(f"Validation images shape: {validation_images.shape}")
            print(f"Validation images min: {validation_images.min():.4f}, max: {validation_images.max():.4f}")
            print(f"Validation images mean: {validation_images.mean():.4f}")

        # We still need the first 4 images for the ShowOxels callback
        show_oxels = ShowOxels(validation_images, every_n_epochs=2, caption="Validation Oxels")
        callbacks.append(show_oxels)

        # Initialize WandB run
        run = wandb.init(
            entity="kl_divergence-rensselaer-polytechnic-institute",
            project="synthetic_3d",
            config=model_config | trainer_config,
            dir="wandb_results",
            mode="offline",
        )

        logger = WandbLogger(
            experiment=run,
        )

        logger.experiment.summary["num_parameters"] = num_parameters
        logger.experiment.summary["effective_batch_size"] = (
            model_config["train_batch_size"] * trainer_config["accumulate_grad_batches"] * get_world_size()
        )
        print(f"Run Id: {logger.experiment.id} Run Name: {logger.experiment.name}")
    else:
        # For non-rank 0 processes, use disabled mode
        logger = WandbLogger(
            name="synthetic",
            project="synthetic_3d",
            save_dir="logs",
            id="synthetic_3d_fixed_v1.0",
            entity="kl_divergence-rensselaer-polytechnic-institute",
            config=model_config | trainer_config,
            settings=wandb.Settings(init_timeout=1800),
            mode="disabled",
        )

    trainer = L.Trainer(**trainer_config, callbacks=callbacks, logger=logger)

    try:
        trainer.fit(model, ckpt_path=ckpt_path)
        if get_rank() == 0:
            result = trainer.callback_metrics.get("training/loss")
            if result is not None:
                logger.experiment.summary["result"] = result
    finally:
        if get_rank() == 0:
            # Clean up WandB
            wandb.finish()


def main(args):
    setup()
    model_config, trainer_config = get_config()
    train(model_config, trainer_config)
    return 0


if __name__ == "__main__":
    import sys
    exit(main(sys.argv)) 