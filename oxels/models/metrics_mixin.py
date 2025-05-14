from lightning import LightningModule


class MetricsMixin(LightningModule):
    def compute_loss(self, batch):
        raise NotImplementedError

    def compute_metrics(self, batch):
        return {"loss": self.compute_loss(batch)}

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics(batch)
        for key, value in metrics.items():
            self.log(f"training/{key}", value, sync_dist=True)

        return metrics["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics(batch)
        for key, value in metrics.items():
            self.log(f"validation/{key}", value, sync_dist=True)

        return metrics["loss"]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics(batch)
        for key, value in metrics.items():
            self.log(f"testing/{key}", value, sync_dist=True)

        return metrics["loss"]
