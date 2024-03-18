from zap.callbacks import Callback
from zap import trainer
from zap.dist import only_master_process, is_distributed

class UnfreezeModel(Callback):
    """
    Unfreeze model parameters after some epoch, by setting requires_grad to True.
    Make sure that the parameters we want to unfreeze are passed to the optimizer. 
    Otherwise, setting requires_grad to True won't matter.

    Args:
        epoch (int) : The epoch at which we want to unfreeze the parameters.
    """
    def __init__(self, epoch : int):
        self.epoch = epoch

    def on_epoch_start(self, trainer : "trainer.Trainer"):
        if trainer.trainer_state.epoch == self.epoch:
            model = (trainer.model.module if is_distributed() else trainer.model)
            for p in model.parameters():
                p.requires_grad = True

class EarlyStopping(Callback):
    def __init__(self, metric_to_track : str, mode : str = "max", patience : int = 10):
        self.metric_to_track = metric_to_track
        self.patience = patience

        # Set up tracking metrics
        self.epochs_since_best = 0
        self.best_metric = None

    def on_validation_end(self, trainer : "trainer.Trainer", val_metrics, Dict[str, Any]):
        current_metric = val_metrics[self.metric_to_track]
        if (
            self.best_metric is None or
            (self.mode == "max" and (current_metric > self.best_metric)) or 
            (self.mode == "min" and (current_metric < self.best_metric))
        ):
            self.best_metric = current_metric
            self.epochs_since_best = 0
        else:
            self.epochs_since_best += 1

        if self.epochs_since_best > self.patience:
            if trainer.is_master_process:
                print(f"It has been {self.epochs_since_best} epochs since the model's {self.metric_to_track} has improved. Stopping training.")
            break

            
            
        
        
    

