from typing import Any, Union, Dict
import os
from pathlib import Path
import torch
#from zap import trainer
from zap.callbacks import Callback
from zap.dist import only_master_process, is_distributed


class DefaultCheckpointer(Callback):
    """
    Saves the following information every `save_freq` epochs:
        * model state dict
        * lr scheduler state dict
        * optimizer state dict
        * epoch
        * global step

    Args:
        checkpoint_file (Union[str, Path]): Where to save the checkpoint.
        save_freq (int) : How often a checkpoint is saved.
    """
    def __init__(self, checkpoint_file : Union[str, Path] = "checkpoint.pt", save_freq : int = 1):
        self.checkpoint_file = Path(checkpoint_file)
        self.save_freq = save_freq

    @only_master_process
    def on_epoch_end(self, trainer : "trainer.Trainer"):
        trainer_state_dict = {}
        if trainer.trainer_state.epoch % self.save_freq == 0:
            model = trainer.model.module if is_distributed() else trainer.model

            trainer_state_dict["model"] = model.state_dict()
            if trainer.lr_scheduler is not None:
                trainer_state_dict["lr_scheduler"] = trainer.lr_scheduler.state_dict()
            trainer_state_dict["optimizer"] = trainer.optimizer.state_dict()
            trainer_state_dict["epoch"] = trainer.trainer_state.epoch
            trainer_state_dict["global_step"] = trainer.trainer_state.global_step

            torch.save(trainer_state_dict, self.checkpoint_file)
