from lightning import LightningModule
import wandb


class MetricsMixin(LightningModule):
    def compute_loss(self, batch):
        raise NotImplementedError

    def compute_metrics(self, batch):
        return {"loss": self.compute_loss(batch)}

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics(batch)
        data = {f"training/{key}": value for key, value in metrics.items()}
        wandb.log(data=data)

        return metrics["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics(batch)
        data = {f"validation/{key}": value for key, value in metrics.items()}
        wandb.log(data=data)

        return metrics["loss"]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics(batch)
        data = {f"testing/{key}": value for key, value in metrics.items()}
        wandb.log(data=data)

        return metrics["loss"]
