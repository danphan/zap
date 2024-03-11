import numpy as np
import torch
from torch import nn
from typing import Optional

def compute_grad_norm(model : nn.Module) -> float:
    total_square_norm = 0
    for p in model.parameters():
        # include param gradient if it's finite
        if p.grad is not None and (torch.isinf(p.grad).sum() == 0):
            total_square_norm += (p.grad**2).sum()

    return total_square_norm.item() ** 0.5

class AutoClip:
    """
    A class to perform automatic gradient clipping based off the model's historical gradients.
    This code was taken from https://github.com/pseeth/autoclip, which is 
    associated with the paper at https://arxiv.org/abs/2007.14469.

    Args:
        percentile (Optional[float]) : the percentile at which the gradient norm 
            is chosen for clipping. If percentile is set to None, no gradient 
            clipping is performed. Default percentile is 10.
    """
    def __init__(self, percentile : Optional[float] = 10):
        self.grad_history = []
        self.percentile = percentile

    def clip(self, model):
        if self.percentile is not None:
            grad_norm = compute_grad_norm(model)
            self.grad_history.append(grad_norm)
            clip_value = np.percentile(self.grad_history, self.percentile)
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
