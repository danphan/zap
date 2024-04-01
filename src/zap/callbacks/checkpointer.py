from typing import Any, Union, Dict
import os
from pathlib import Path
import torch
from zap import trainer
from zap.callbacks import Callback
from zap.dist import only_master_process, is_distributed

def save_checkpoint(trainer : "trainer.Trainer", checkpoint_file):
    """
    Saves the following information from the trainer:
        * model state dict
        * lr scheduler state dict
        * optimizer state dict
        * epoch
        * global step
    """
    trainer_state_dict = {}
    model = trainer.model.module if is_distributed() else trainer.model

    trainer_state_dict["model"] = model.state_dict()
    if trainer.lr_scheduler is not None:
        trainer_state_dict["lr_scheduler"] = trainer.lr_scheduler.state_dict()
    trainer_state_dict["optimizer"] = trainer.optimizer.state_dict()
    trainer_state_dict["epoch"] = trainer.trainer_state.epoch
    trainer_state_dict["global_step"] = trainer.trainer_state.global_step

    torch.save(trainer_state_dict, checkpoint_file)


class DefaultCheckpointer(Callback):
    """
    Checkpoints every `save_freq` epochs:

    Args:
        checkpoint_file (Union[str, Path]): Where to save the checkpoint.
        save_freq (int) : How often a checkpoint is saved.
    """
    def __init__(self, checkpoint_file : Union[str, Path] = "checkpoint.pt", save_freq : int = 1):
        self.checkpoint_file = Path(checkpoint_file)
        self.save_freq = save_freq

    @only_master_process
    def on_epoch_end(self, trainer : "trainer.Trainer"):
        if trainer.trainer_state.epoch % self.save_freq == 0:
            save_checkpoint(trainer, self.checkpoint_file)

class BestAndLastCheckpointer(Callback):
    """
    Checkpoints the trainer every epoch as "last.pt". Based on a given validation metric,
    a "best.pt" will also be saved.

    Args:
        metric_to_track (str) : The name of the validation metric to track (must be a key in the Trainer's val_metrics)
        mode (str) : One of "min", "max". Default: "max".
    """
    def __init__(self, metric_to_track : str, mode : str = "max", checkpoint_dir : Union[str, Path] = "."):
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents = True, exist_ok = True)

        self.best_checkpoint_path = self.checkpoint_dir / "best.pt"
        self.last_checkpoint_path = self.checkpoint_dir / "last.pt"
        if mode not in ["min", "max"]:
            raise ValueError('mode must be either "min" or "max"')
        self.mode = mode
        self.metric_to_track = metric_to_track
        self.best_metric = None

    @only_master_process
    def on_epoch_end(self, trainer : "trainer.Trainer"):
        save_checkpoint(trainer, self.last_checkpoint_path)

    @only_master_process
    def on_validation_end(self, trainer : "trainer.Trainer", val_metrics : Dict[str, Any]):
        current_metric = val_metrics[self.metric_to_track]
        if (
            self.best_metric is None or
            (self.mode == "max" and (current_metric > self.best_metric)) or 
            (self.mode == "min" and (current_metric < self.best_metric))
        ):
            self.best_metric = current_metric
            save_checkpoint(trainer, self.best_checkpoint_path)

            
            
        
        
    

