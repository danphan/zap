import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
import math
import matplotlib.pyplot

def linear_warmup_cosine_annealing_scheduler(
    optimizer : Optimizer,
    min_lr : float, 
    max_lr : float,
    total_steps : int,
    pct_start : float = 0.3,
    final_div_factor : float = 10000.0,    
) -> LRScheduler:

    lr_final = max_lr / final_div_factor

    def learning_rate_fn(step) -> float:

        # Number of steps it takes to go from min_lr to max_lr during warmup
        num_warmup_steps = int(pct_start * total_steps)

        # Number of annealing steps (going from max_lr to max_lr / final_div_factor)
        num_annealing_steps = total_steps - num_warmup_steps

        if step < num_warmup_steps:
            lr = min_lr + step * (max_lr - min_lr)  / (num_warmup_steps - 1)
        else:
            step = step - num_warmup_steps
            lr = lr_final + (max_lr - lr_final) / 2 * (1 + math.cos(math.pi * step / (num_annealing_steps - 1)))

        return lr  / min_lr
    
#    x = np.arange(total_steps)
#    y = [learning_rate_fn(step) for step in x]
#
#    plt.plot(x,y)
#    plt.savefig("test.png")
    
    return LambdaLR(optimizer = optimizer, lr_lambda = learning_rate_fn)

if __name__ == "__main__":
    from torch.optim import Adam 
    import torch.nn as nn

    min_lr = 1e-4
    max_lr = 1e-3

    total_steps = 1000
    pct_start = 0.05
    

    model = nn.Linear(1,1)
    optimizer = Adam(model.parameters(), lr = 1)

    scheduler = linear_warmup_cosine_annealing_scheduler(
        optimizer=optimizer, 
        min_lr = min_lr, 
        max_lr = max_lr, 
        total_steps = total_steps,
        pct_start=pct_start,
    )
