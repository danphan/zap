from typing import Callable, Dict, List
import torch
import torch.distributed as dist
from zap.dist import is_distributed

class LossTracker:
    """
    A torchmetrics.Metric-like API for tracking the loss in a distributed setting.

    Args:
        loss_fn (Callable) : A Callable which takes in preds and labels and returns Dict[str,torch.Tensor],
           (this should be model.loss)
    """

    def __init__(
        self, 
        model,
    ):
        self.model = model
        self.total_loss = {name : torch.tensor(0.) for name in self.model.loss_names}
        self.num_samples = 0

    def update(self, preds : torch.Tensor, labels : torch.Tensor) -> None:
        """
        Updates self.total_loss and self.num_samples
        """
        self.total_loss = {name : loss.to(preds.device) for name, loss in self.total_loss.items()}

        batch_losses = self.model.loss(preds, labels)
        batch_size = preds.shape[0]

        for name in self.model.loss_names:
            if is_distributed():
                # Add all losses (use all_reduce just in case)
                dist.all_reduce(batch_losses[name], dist.ReduceOp.SUM)
                #dist.reduce(batch_losses[name], 0, dist.ReduceOp.SUM)

            self.total_loss[name] += batch_losses[name] * batch_size # unnormalize the loss, since it is defined per-sample

        self.num_samples += batch_size

    def compute(self) -> Dict[str, torch.Tensor]:
        return {name : loss / self.num_samples for name, loss in self.total_loss.items()}
            
            
