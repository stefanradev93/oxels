from lightning import LightningModule


class MetricsMixin(LightningModule):
    def compute_loss(self, batch):
        raise NotImplementedError

    def compute_metrics(self, batch):
        return {"loss": self.compute_loss(batch)}

    def compute_metrics_validation(self, batch):
        return {"loss": self.compute_loss_validation(batch)}

    def compute_metrics_test(self, batch):
        return {"loss": self.compute_loss_test(batch)}

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics(batch)
        data = {f"training/{key}": value for key, value in metrics.items()}
        for key, value in data.items():
            self.log(key, value, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        return metrics["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics_validation(batch)
        data = {f"validation/{key}": value for key, value in metrics.items()}
        for key, value in data.items():
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        return metrics["loss"]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics_test(batch)
        data = {f"testing/{key}": value for key, value in metrics.items()}
        for key, value in data.items():
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        return metrics["loss"]
