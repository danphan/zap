from typing import Optional, Any, Dict, Callable, List
from dataclasses import dataclass
from tqdm import tqdm
import warnings
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import torch 
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, ReduceLROnPlateau
#from .logger import Logger
from zap.callbacks import Callback
from zap.autoclip import AutoClip

@dataclass
class TrainerState:
    epoch : int = 0
    global_step : int = 0



class Trainer:
    def __init__(
        self, 
        model : nn.Module, 
        train_dataloader : DataLoader,
        val_dataloader : DataLoader,
        val_metrics : Dict[str, Callable],
        optimizer : Optimizer,
        #scheduler : LRScheduler = None,
        use_amp : bool = True,
        device : str = "cuda",
        #loss_file : Union[str,Path] = "train_loss.csv",
        #metrics_file : Union[str,Path] = "val_metrics.csv",
        #checkpoint_dir : str = ".",
        autoclip_percentile : Optional[float] = 10.0,
        #update_lr_every : str = "epoch",
        #metric_to_track : Optional[str] = None,
        callbacks : Optional[List[Callback]] = None,
    ) -> None:
        """
        Args:
            model (nn.Module) : During training, the model should output a dictionary of losses 
                during training for each batch. During inference, the model should output the predictions.
                The model should have :
                    * a method `loss()`, which takes model outputs and returns the dictionary
                    of losses.
                    * an attribute `loss_names` which is the list of names for each loss component.


            val_metrics (Dict[str, Callable]) : the callbacks one should run during validation. Each Callable should take in a batch 
                and output floats.

            autoclip_percentile (Optional[float]) : the percentile to use for autoclipping, in the range [0,100]. 
                If 0, the model is never updated. If 100, there is no gradient clipping. If this parameter is set to None,
                no gradient clipping is performed (equivalent to 100, but with less overhead.) Default: 10.

            update_lr_every (str) : The frequency that the learning rate is updated (one of "batch" or "epoch").
                If scheduler is None, the learning rate is never updated, and this variable is ignored.
            metric_to_track (Optional[str]) : The validation metric to track. Only used 
                if using ReduceLROnPlateau as the learning rate scheduler.

            checkpoint_dir (str) : the path to the directory where the checkpoints should be stored. By default,
                In this directory, we store the model with the best validation metrics (see metric_to_track).

        """

        # Model
        self.model = model
        self.device = device
        self.device_type = device.split(":")[0] # "cuda" or "cpu"
        self.model.to(device)

        # Training/validation data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimization 
        self.optimizer = optimizer
        #self.scheduler = scheduler

        # Update learning rate every epoch or batch
        #assert update_lr_every in ["batch", "epoch"], "update_lr_every must be either 'batch' or 'epoch'"
        #self.update_lr_every = update_lr_every

        ## Handle case when using ReduceLROnPlateau as learning rate scheduler
        #self.metric_to_track = metric_to_track
        #if isinstance(self.scheduler, ReduceLROnPlateau):
        #    assert metric_to_track in val_metrics.keys(), "metric_to_track must be one of the keys in val_metrics"

        #    if self.update_lr_every == "batch":
        #        warnings.warn("ReduceLROnPlateau as implemented only works if the learning rate is updated per epoch. Resetting update_every_lr = 'batch'.")
        #        self.update_lr_every = "epoch"


        # Automatic mixed precision
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

        ## Validation metrics/logging
        self.val_metrics = val_metrics
        #self.loss_logger = Logger(loss_file)
        #self.val_logger = Logger(metrics_file)
#
#        self.checkpoint_dir = Path(checkpoint_dir)

        # For gradient clipping
        self.autoclipper = AutoClip(percentile=autoclip_percentile)

        # Set up callbacks
        self.callbacks = callbacks if (callbacks is not None) else []

        # Set up trainer state
        self.trainer_state = TrainerState() # initialize epoch and global_step to 0.

    def run_callbacks(self, callback_method : str, *args, **kwargs):
        """
        Loop through the callbacks and run the appropriate methods,
        as specified via `callback_location`.

        Args:
            callback_method (str) : e.g. "on_train_start", "on_validation_end", etc.
                For a full list, see the definition of Callback.
        """

        for callback in self.callbacks:
            method = getattr(callback, callback_method)
            method(*args, **kwargs)


    def fit(self, num_epochs : int) -> None:
        self.run_callbacks("on_train_start", self)

        for epoch in range(num_epochs):
            # Train
            self.train_one_epoch(num_epochs)

            self.run_callbacks("on_epoch_end", self)

            # Validate
            val_metrics = self.evaluate()

            self.run_callbacks("on_validation_end", self, val_metrics)

            self.trainer_state.epoch += 1

#            # Log metrics
#            self.val_logger.log(val_metrics)
#
#            # Update learning rate
#            if self.update_lr_every == "epoch":
#                if isinstance(self.scheduler, ReduceLROnPlateau):
#                    self.scheduler.step(val_metrics[self.metric_to_track])
#                elif self.scheduler is not None:
#                    self.scheduler.step()
#
#            # Store checkpoint
#            self.checkpoint()



    def train_one_step(self, batch : Dict[str, Any], batch_idx : int) -> Dict[str, torch.Tensor]:
        """
        Performs one training step. Returns dictionary of losses for logging.
        """

        batch = {k : v.to(self.device) for k, v in batch.items()}

        # Forward pass
        with torch.autocast(device_type = self.device_type, dtype = torch.float16):
            losses : Dict[str, torch.Tensor] = self.model(**batch)
            total_loss = sum(losses.values())
            losses.update(total_loss = total_loss)

        # Backward pass
        self.scaler.scale(total_loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.optimizer)

        # Clip the gradients, automatically determining an appropriate clipping value.
        self.autoclipper.clip(self.model)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them. 
        # update weights (if there aren't inf's or NaNs in grads. if so, no step is taken.)
        self.scaler.step(self.optimizer)

        # Update scale
        self.scaler.update()

        # Zero-out gradients
        self.optimizer.zero_grad()

        ## Update learning rate if using a scheduler
        #if (self.scheduler is not None) and (self.update_lr_every == "batch"):
        #    self.scheduler.step()


        return losses

    def train_one_epoch(self, num_epochs):
        self.model.train()

        description = ("\n" + "{:20s}"*(3 + len(self.model.loss_names))).format(
            "Epoch",
            "GPU_mem",
            *self.model.loss_names,
            "total_loss",
        )
        print(description)
        pbar = tqdm(enumerate(self.train_dataloader), total = len(self.train_dataloader))
        for batch_idx, batch in pbar:
            losses = self.train_one_step(batch, batch_idx)

            self.run_callbacks("on_step_end", self, losses)

            self.trainer_state.global_step += 1
            #self.loss_logger.log(losses)
            mem = f"{torch.cuda.memory_reserved(device = self.device)/1e9 if torch.cuda.is_available() else 0:.3g}G" # GB
            pbar.set_description(
                ("{:20s}" + "{:20s}" + "{:<20.3g}"*len(losses.values())).format(
                    f"{self.trainer_state.epoch + 1}/{num_epochs}",
                    mem,
                    *losses.values(),
                )
            )

#    def checkpoint(self):
#        """ Store model checkpoint"""
#        torch.save(self.model.state_dict(), self.checkpoint_dir / "last.pt")
        


    def evaluate(self) -> Dict[str, float]:
        """
        Perform validation and return metrics (possibly for use with LR scheduling.)
        """
        self.model.eval()

        # Set up dictionary of metric values
        val_losses = {loss_name : 0. for loss_name in self.model.loss_names}
        metrics = {metric_name : 0. for metric_name in self.val_metrics.keys()}

        num_metrics = len(val_losses) + len(metrics)

        num_samples = 0

        print(("{:20s}" * num_metrics).format(*val_losses.keys(), *metrics.keys()))
        pbar = tqdm(self.val_dataloader)
        for batch in pbar:
            batch = {k : v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                preds = self.model(**batch)
            batch_size = preds.shape[0]
            labels = batch["label"]

            batch_metrics = {metric_name : metric_fn(preds, labels) for metric_name, metric_fn in self.val_metrics.items()}
            batch_losses = self.model.loss(preds, labels)

            for k, v in batch_metrics.items():
                metrics[k] = (metrics[k] * num_samples + v * batch_size) / (num_samples + batch_size)

            for k, v in batch_losses.items():
                val_losses[k] = (val_losses[k] * num_samples + v * batch_size) / (num_samples + batch_size)

            num_samples += batch_size

            pbar.set_description(
                ("{:<20.3g}" * num_metrics).format(*val_losses.values(), *metrics.values())
            )

        metrics = val_losses | metrics

        return metrics

    def make_lr_finder_dataloader(
        self,
        num_iter : int,
        batch_size : int,
    ) -> DataLoader:

        dataset = self.train_dataloader.dataset

        sampler = RandomSampler(
            data_source = dataset,
            replacement = True,
            num_samples = batch_size * num_iter,
        )

        return DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = sampler,
            num_workers = self.train_dataloader.num_workers,
            pin_memory = self.train_dataloader.pin_memory,
        )
            

    
    def find_lr(
        self, 
        min_lr : float = 1e-8, 
        max_lr : float = 1, 
        num_iter : int = 1000,
        momentum : float = 0.98, 
        loss_file : str = "lr_finder.csv", 
        plot_path : str = "lr_finder.png",
    ):
        """
        Performs the learning rate range test proposed by Smith in https://arxiv.org/pdf/1506.01186.pdf.

        This will create a new dataloader so that we can iterate through any number of batches,
        as opposed to stopping when we reach the end of the dataset. 

        Args:
            min_lr (float) : The minimum learning rate.
            max_lr (float) : The maximum learning rate.
            num_iter (int) : the number of iterations (i.e. batches) to go through.
            momentum (float) : The momentum to use for the exponential moving average (must be between 0 and 1.)
            loss_file (str) : The file in which the losses will be logged.
            plot_path (str) : The output file for the loss vs learning-rate plot.
        """

        loader = self.make_lr_finder_dataloader(num_iter, self.train_dataloader.batch_size)

#        if self.scheduler is not None:
#            warnings.warn("Learning rate schedulers input during initialization are ignored (and overwritten) when using this method!")

        mult_factor = (max_lr / min_lr) ** (1 / (len(loader) - 1))

        # Reinitialize the optimizer using min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = min_lr
            param_group["initial_lr"] = min_lr

        scheduler = LambdaLR(
            optimizer = self.optimizer,
            lr_lambda = lambda iter_idx : mult_factor ** iter_idx
        )

        avg_loss = 0
        smoothed_loss = 0
        best_loss = 0

        lr_list = []
        loss_list = []

        with open(loss_file,"w") as f:
            f.write("lr,loss\n")

        iter_idx = 0

        for batch in tqdm(loader):
            batch_losses = self.train_one_step(batch)

            # Plot losses
            total_batch_loss = batch_losses["total_loss"]
            #total_batch_loss = sum(batch_losses.values())
            avg_loss = momentum * avg_loss + (1 - momentum) * total_batch_loss
            smooth_loss = avg_loss / (1 - momentum ** (iter_idx + 1))

            if iter_idx == 0 or smooth_loss < best_loss:
                best_loss = smooth_loss

            if smooth_loss > 4 * best_loss:
                break

            # Log losses
            with open(loss_file,"a") as f:
                lr = scheduler.get_last_lr()[0]
                f.write(f"{lr},{smooth_loss:.8g}\n")

            lr_list.append(lr)
            loss_list.append(smooth_loss.detach().cpu().numpy().item())

            iter_idx += 1

        plt.semilogx(lr_list, loss_list)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.savefig(plot_path)


                
