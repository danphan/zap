from typing import Any, Union, Dict
import os
from pathlib import Path
#from zap import trainer
from zap.callbacks import Callback
from zap.dist import only_master_process, is_distributed


class DefaultCheckpointer(Callback):
    """
    Saves the model weights every n epochs.
    """
    def __init__(self, save_file : Union[str, Path] = "model.pt", save_freq : int = 1):
        self.save_file = Path(save_file)
        self.save_freq = save_freq

    @only_master_process 
    def on_epoch_end(self, trainer : "trainer.Trainer"):
        if trainer.trainer_state.epoch % self.save_freq == 0:
            if is_distributed():
                torch.save(trainer.model.module.state_dict(), self.save_file)
            else:
                torch.save(trainer.model.state_dict(), self.save_file)

