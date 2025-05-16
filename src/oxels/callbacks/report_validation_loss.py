import lightning as L
import optuna


class ReportValidationLoss(L.Callback):
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict):
            outputs = outputs["loss"]

        self.trial.report(outputs, step=trainer.global_step)

        if self.trial.should_prune():
            raise optuna.TrialPruned()
