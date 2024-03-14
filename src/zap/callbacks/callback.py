from typing import Any, Union, Dict
import os
import functools
from pathlib import Path
from zap import trainer
from zap.logger import Logger
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import torch
from torch import nn
from zap.dist import only_master_process


class Callback:

    def on_train_start(self, trainer : "trainer.Trainer"):
        pass

    def on_step_end(self, trainer : "trainer.Trainer", model_losses : Any):
        """
        Event called after the end of a training step.
        """
        pass

    def on_epoch_end(self, trainer : "trainer.Trainer"):
        """
        Event called after finishing epoch, but before validation.
        """
        pass

    def on_validation_end(self, trainer : "trainer.Trainer", val_metrics : Dict[str, Any]):
        """
        Event called after finishing epoch, but before validation.
        """
        pass

class DefaultCheckpointer(Callback):
    """
    Saves the model weights every n epochs.
    """
    def __init__(self, save_file : Union[str, Path] = "model.pt", save_freq : int = 1):
        self.save_file = Path(save_file)
        self.save_freq = save_freq

    def on_epoch_end(self, trainer : "trainer.Trainer"):
        if trainer.trainer_state.epoch % self.save_freq == 0:
            torch.save(trainer.model.state_dict(), self.save_file)

class DefaultLRSchedulerCallback(Callback):
    """
    Updates the learning rate with some desired frequency (e.g. every n batches, every epoch, etc.)
    """
    def __init__(
        self, 
        scheduler : LRScheduler,
        step_or_epoch : str = "step", 
        update_freq : int = 1,
    ): 
        self.scheduler = scheduler

        assert step_or_epoch in ["step", "epoch"], "`step_or_epoch` must be 'step' or 'epoch'"
        self.step_or_epoch = step_or_epoch
        self.update_freq = update_freq

    def on_step_end(self, trainer, model_losses):
        if self.step_or_epoch == "step" and (trainer.trainer_state.global_step % self.update_freq == 0):
            self.scheduler.step()

    def on_epoch_end(self, trainer):
        if self.step_or_epoch == "epoch" and (trainer.trainer_state.epoch % self.update_freq == 0):
            self.scheduler.step()

class ReduceLROnPlateauCallback(Callback):
    def __init__(
        self,
        scheduler : ReduceLROnPlateau,
        metric_to_track : str,
    ):
        assert isinstance(scheduler, ReduceLROnPlateau), "LR scheduler needs to be an instance of ReduceLROnPlateau"
        self.scheduler = scheduler
        self.metric_to_track = metric_to_track

    def on_train_start(self, trainer):
        if self.metric_to_track not in trainer.val_metrics.keys():
            raise KeyError(
                f"{self.metric_to_track} is not being evaluated during validation and therefore cannot be tracked. Tracked metrics: {self.val_metrics.keys()}"
            )
    def on_validation_end(self, trainer, val_metrics):
        self.scheduler.step(val_metrics[self.metric_to_track])

class TrainLossLogger(Callback):
    def __init__(
        self,
        loss_file : Union[str, Path] = "train_loss.csv",
        save_freq : int = 1
    ):
        self.logger = Logger(loss_file)
        self.save_freq = save_freq


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

    def on_validation_end(self, trainer, val_metrics):
        loss_logs = {
            "epoch" : trainer.trainer_state.epoch,
            "global_step" : trainer.trainer_state.global_step,
        } | val_metrics
        self.logger.log(val_metrics)
            

