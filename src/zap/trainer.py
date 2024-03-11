from typing import Optional
from tqdm import tqdm
import warnings
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import torch 
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from typing import Dict, Callable
from .logger import Logger
from .autoclip import AutoClip


class Trainer:
    def __init__(
        self, 
        model : nn.Module, 
        train_dataloader : DataLoader,
        val_dataloader : DataLoader,
        val_metrics : Dict[str, Callable],
        optimizer : Optimizer = None,
        scheduler : LRScheduler = None,
        use_amp : bool = True,
        device : str = "cuda",
        loss_logger : Logger = None,
        val_logger : Logger = None,
        checkpoint_file : str = "model_weights.pt",
        autoclip_percentile : Optional[float] = 10.0,
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
        self.scheduler = scheduler

        # Automatic mixed precision
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

        # Validation metrics/logging
        self.val_metrics = val_metrics
        self.loss_logger = loss_logger
        self.val_logger = val_logger

        self.checkpoint_file = checkpoint_file

        # For gradient clipping
        self.autoclipper = AutoClip(percentile=autoclip_percentile)


    def fit(self, num_epochs : int) -> None:
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch,num_epochs)
            self.evaluate()

        # Store checkpoint
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def train_one_step(self, batch : dict) -> dict:
        """
        Performs one training step. Returns dictionary of losses for logging.
        """

        batch = {k : v.to(self.device) for k, v in batch.items()}

        # Forward pass
        with torch.autocast(device_type = self.device_type, dtype = torch.float16):
            losses : Dict[str, torch.Tensor]= self.model(**batch)
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

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        return losses

    def train_one_epoch(self, epoch, num_epochs):
        self.model.train()

        description = ("\n" + "{:20s}"*(3 + len(self.model.loss_names))).format(
            "Epoch",
            "GPU_mem",
            *self.model.loss_names,
            "total_loss",
        )
        print(description)
        pbar = tqdm(self.train_dataloader)
        for batch in pbar:
            losses = self.train_one_step(batch)
            #self.loss_logger.log({k : float(v) for k, v in losses.items()})
            self.loss_logger.log(losses)
            mem = f"{torch.cuda.memory_reserved(device = self.device)/1e9 if torch.cuda.is_available() else 0:.3g}G" # GB
            pbar.set_description(
                ("{:20s}" + "{:20s}" + "{:<20.3g}"*len(losses.values())).format(
                    f"{epoch + 1}/{num_epochs}",
                    mem,
                    *losses.values(),
                )
            )




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

        self.val_logger.log(metrics)

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

        if self.scheduler is not None:
            warnings.warn("Learning rate schedulers input during initialization are ignored (and overwritten) when using this method!")

        mult_factor = (max_lr / min_lr) ** (1 / (len(loader) - 1))

        # Reinitialize the optimizer using min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = min_lr
            param_group["initial_lr"] = min_lr

        self.scheduler = LambdaLR(
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
                lr = self.scheduler.get_last_lr()[0]
                f.write(f"{lr},{smooth_loss:.8g}\n")

            lr_list.append(lr)
            loss_list.append(smooth_loss.detach().cpu().numpy().item())

            iter_idx += 1

        plt.semilogx(lr_list, loss_list)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.savefig(plot_path)


                
