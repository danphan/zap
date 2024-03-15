from pathlib import Path
from typing import Any, Union, Dict
import os
from zap import trainer
from zap.callbacks.callback import Callback
from zap.dist import only_master_process

class Logger:
    def __init__(self, log_file : Union[str,Path]):
        self.log_file = Path(log_file)

    def log(self, metrics : Dict[str,float]):
        if not self.log_file.is_file():
            with open(self.log_file, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
        with open(self.log_file, "a") as f:
            f.write(",".join([f"{v:<4.4g}" for v in metrics.values()]) + "\n")

class TrainLossLogger(Callback):
    def __init__(
        self,
        loss_file : Union[str, Path] = "train_loss.csv",
        save_freq : int = 1
    ):
        self.logger = Logger(loss_file)
        self.save_freq = save_freq


    @only_master_process 
    def on_step_end(self, trainer, model_losses):
        if trainer.trainer_state.global_step % self.save_freq == 0:
            loss_logs = {
                "epoch" : trainer.trainer_state.epoch,
                "global_step" : trainer.trainer_state.global_step,
                } | model_losses
            self.logger.log(loss_logs)

class ValidationMetricLogger(Callback):
    def __init__(
        self,
        metric_file : Union[str, Path] = "val_metrics.csv",
    ):
        self.logger = Logger(metric_file)

    @only_master_process 
    def on_validation_end(self, trainer, val_metrics):
        loss_logs = {
            "epoch" : trainer.trainer_state.epoch,
            "global_step" : trainer.trainer_state.global_step,
        } | val_metrics
        self.logger.log(val_metrics)
            

