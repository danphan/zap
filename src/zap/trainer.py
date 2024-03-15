from typing import Optional, Any, Dict, Callable, List
import os
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

import torch 
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torchmetrics import Metric
from torch.nn.parallel import DistributedDataParallel as DDP

from zap.callbacks import Callback
from zap.autoclip import AutoClip
from zap.metrics import LossTracker
from zap.dist import is_distributed
from zap.sampler import DistributedRandomSampler

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
        optimizer : Optimizer,
        lr_scheduler : Optional[LRScheduler] = None,
        val_metrics : Optional[Dict[str, Metric]] = None,
        use_amp : bool = True,
        autoclip_percentile : Optional[float] = 10.0,
        callbacks : Optional[List[Callback]] = None,
        device : Optional[int] = None,
    ) -> None:
        """
        Args:
            model (nn.Module) : During training, the model should output a dictionary of losses 
                during training for each batch. During inference, the model should output the predictions.
                The model should have :
                    * a method `loss()`, which takes model outputs and returns the dictionary
                    of losses.
                    * an attribute `loss_names` which is the list of names for each loss component.


            #val_metrics (Dict[str, Callable]) : the callbacks one should run during validation. Each Callable should take in a batch 
            #    and output floats.
            val_metrics (Dict[str, Metric]) : the callbacks one should run during validation. Each Callable should take in a batch 
                and output floats.

            autoclip_percentile (Optional[float]) : the percentile to use for autoclipping, in the range [0,100]. 
                If 0, the model is never updated. If 100, there is no gradient clipping. If this parameter is set to None,
                no gradient clipping is performed (equivalent to 100, but with less overhead.) Default: 10.

        """

        # Model
        self.model = model

        if is_distributed():
            if device is not None:
                raise ValueError("device should not be set when running a distributed job.")
            self.device = int(os.environ["LOCAL_RANK"])
            self.model.to(self.device)
            self.model = DDP(self.model, device_ids = [self.device])
        else:
            self.device = device
            self.model.to(self.device)

        # Parallelism setup
        self.rank = int(os.getenv("RANK",-1))
        self.is_master_process = (self.rank < 1) #single-GPU jobs are the master process by default.

        # Training/validation data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimization 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Automatic mixed precision
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

        ## Validation metrics/logging
        if val_metrics is not None:
            self.val_metrics = {metric_name : metric.to(self.device) for metric_name, metric in val_metrics.items()}
        else:
            self.val_metrics = None

        # For gradient clipping
        self.autoclipper = None if (autoclip_percentile is None) else AutoClip(autoclip_percentile)

        # Set up callbacks
        self.callbacks = callbacks if (callbacks is not None) else []

        # Set up trainer state
        self.trainer_state = TrainerState() # initialize epoch and global_step to 0.

        # Set up tracker for validation loss 
        self.val_loss_tracker = LossTracker(self.model.module if is_distributed() else self.model)

        self.loss_names = self.model.module.loss_names if is_distributed() else self.model.loss_names

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

            # Reset sampler if training in a distributed fashion to ensure variation in shuffling
            if is_distributed():
                self.train_dataloader.sampler.set_epoch(epoch)

            # Train
            self.train_one_epoch(num_epochs)

            self.run_callbacks("on_epoch_end", self)

            # Validate
            if self.val_metrics is not None:
                val_metrics = self.evaluate()

                self.run_callbacks("on_validation_end", self, val_metrics)

            self.trainer_state.epoch += 1

    def train_one_step(self, batch : Dict[str, Any], batch_idx : int) -> Dict[str, torch.Tensor]:
        """
        Performs one training step. Returns dictionary of losses for logging.
        """

        batch = {k : v.to(self.device) for k, v in batch.items()}

        # Forward pass
        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            losses : Dict[str, torch.Tensor] = self.model(**batch)
            total_loss = sum(losses.values())
            losses.update(total_loss = total_loss)

        # Backward pass
        self.scaler.scale(total_loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.optimizer)

        # Clip the gradients, automatically determining an appropriate clipping value.
        if self.autoclipper is not None:
            self.autoclipper.clip(self.model)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them. 
        # update weights (if there aren't inf's or NaNs in grads. if so, no step is taken.)
        self.scaler.step(self.optimizer)

        # Update scale
        self.scaler.update()

        # Zero-out gradients
        self.optimizer.zero_grad()

        return losses

    def train_one_epoch(self, num_epochs):
        self.model.train()

        description = ("\n" + "{:20s}"*(3 + len(self.loss_names))).format(
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "total_loss",
        )

        if self.is_master_process:
            print(description)

        pbar = tqdm(enumerate(self.train_dataloader), total = len(self.train_dataloader), disable = not self.is_master_process)

        for batch_idx, batch in pbar:
            # Perform one optimization step
            losses = self.train_one_step(batch, batch_idx)

            # Avg losses across processes
            if is_distributed():
                for name in self.loss_names:
                    torch.distributed.all_reduce(losses[name], torch.distributed.ReduceOp.AVG)

            self.run_callbacks("on_step_end", self, losses)

            self.trainer_state.global_step += 1

            mem = f"{torch.cuda.memory_reserved(device = self.device)/1e9 if torch.cuda.is_available() else 0:.3g}G" # GB

            pbar.set_description(
                ("{:20s}" + "{:20s}" + "{:<20.3g}"*len(losses.values())).format(
                    f"{self.trainer_state.epoch + 1}/{num_epochs}",
                    mem,
                    *losses.values(),
                )
            )

        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


    def evaluate(self) -> Dict[str, float]:
        """
        Perform validation and return metrics.
        """
        self.model.eval()

        # Reset state of loss tracker/metric to avoid contamination between epochs
        self.val_loss_tracker.clear_state()
        for metric in self.val_metrics.values():
            metric.reset()

        num_metrics = len(self.loss_names) + len(self.val_metrics)

        if self.is_master_process:
            print(("{:20s}" * num_metrics).format(*self.loss_names, *self.val_metrics.keys()))
        pbar = tqdm(self.val_dataloader, disable = not self.is_master_process)
        for batch in pbar:
            batch = {k : v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                preds = self.model(**batch)
            batch_size = preds.shape[0]
            labels = batch["label"]

            # Calculate running metric
            running_metrics = {}
            for name, metric in self.val_metrics.items():
                metric.update(preds, labels)
                running_metrics[name] = metric.compute()

            # Calculate running losses
            self.val_loss_tracker.update(preds, labels)
            running_losses = self.val_loss_tracker.compute()

            pbar.set_description(
                ("{:<20.3g}" * num_metrics).format(*running_losses.values(), *running_metrics.values())
            )

        return running_metrics | running_losses

    def make_lr_finder_dataloader(
        self,
        num_iter : int,
        batch_size : int,
    ) -> DataLoader:

        dataset = self.train_dataloader.dataset

        if is_distributed():
            sampler = DistributedRandomSampler(
                data_source = dataset,
                num_samples = batch_size * num_iter,
            )
        else:
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
        num_iter : int = 1000,
        min_lr : float = 1e-8, 
        max_lr : float = 1, 
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

        for batch_idx, batch in tqdm(enumerate(loader), total = num_iter, disable = not self.is_master_process):
            batch_losses = self.train_one_step(batch, batch_idx)

            # Update learning rate
            scheduler.step()

            # Plot losses
            total_batch_loss = batch_losses["total_loss"]
            #total_batch_loss = sum(batch_losses.values())
            avg_loss = momentum * avg_loss + (1 - momentum) * total_batch_loss
            smooth_loss = avg_loss / (1 - momentum ** (iter_idx + 1))

            if is_distributed():
                torch.distributed.all_reduce(smooth_loss, torch.distributed.ReduceOp.AVG)

            if iter_idx == 0 or smooth_loss < best_loss:
                best_loss = smooth_loss

            if smooth_loss > 4 * best_loss:
                break

            # Log losses
            if self.is_master_process:
                with open(loss_file,"a") as f:
                    lr = scheduler.get_last_lr()[0]
                    f.write(f"{lr},{smooth_loss:.8g}\n")

                lr_list.append(lr)
                loss_list.append(smooth_loss.detach().cpu().numpy().item())

            iter_idx += 1

        if self.is_master_process:
            plt.semilogx(lr_list, loss_list)
            plt.xlabel("Learning rate")
            plt.ylabel("Loss")
            plt.savefig(plot_path)

