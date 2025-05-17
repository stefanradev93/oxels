import os
import json
import torch
import torch.distributed as dist
import optuna
import lightning as L
from lightning.pytorch import Trainer

# Helper to get rank for SLURM/torch.distributed
def get_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        return 0

def get_world_size():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        return 1

def broadcast_object(obj, src=0):
    # Simple broadcast via torch.distributed; must be called after DDP init!
    objstr = json.dumps(obj) if get_rank() == src else ""
    objstr = [objstr]
    if dist.is_initialized():
        dist.broadcast_object_list(objstr, src=src)
    return json.loads(objstr[0])

def objective_hparams():  # No Optuna trial here!
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn
    import torch.nn.functional as F

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

    return MinimalModel

def single_trial(hparams):
    # This runs on all ranks with hparams from rank 0
    Model = objective_hparams()
    model = Model(hparams['dim'])
    trainer = Trainer(
        max_steps=100,
        accelerator="gpu",  # or "cuda"
        devices=get_world_size(), # or "auto"
        num_nodes=1,   # Lightning will auto-detect via the launch
        strategy="ddp",
        precision="16-mixed",
    )
    trainer.fit(model)
    # For demo: just use a random score
    score = torch.rand(1).item() if get_rank() == 0 else None
    return score

def main():
    import optuna

    rank = get_rank()
    world_size = get_world_size()
    ddp = (world_size > 1)
    seed = int(os.environ.get("SLURM_JOB_ID", "0"))
    L.seed_everything(seed)

    # DDP init must happen before broadcast below if using torch.distributed
    if ddp and not dist.is_initialized():
        # Use default backend or read from env as needed
        dist.init_process_group(backend="nccl")

    # --- Optuna trial only on rank 0 ---
    if rank == 0:
        def suggest_params(trial):
            dim = trial.suggest_int("dim", 3, 5)
            return {'dim': dim}

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            study_name="optuna-test",
            storage="sqlite:///optuna-test.db",
            load_if_exists=True,
            sampler=sampler,
            direction="minimize"
        )

        def objective(trial):
            hparams = suggest_params(trial)
            # Broadcast hparams to all ranks
            broadcast_object(hparams, src=0)
            score = single_trial(hparams)
            # You may retrieve validation loss at end of training for the score
            return score

        study.optimize(objective, n_trials=10)
    else:
        # Other ranks: wait for hparams and run training
        for _ in range(10):  # match n_trials above
            hparams = broadcast_object(None, src=0)
            single_trial(hparams)

if __name__ == "__main__":
    main()
