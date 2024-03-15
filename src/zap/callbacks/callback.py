from typing import Any, Union, Dict
from zap import trainer


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

