# Zap

Zap is a barebones Trainer class with some basic callbacks. This is meant to eliminate the boilerplate training code in pytorch, while allowing full customization if needed. 
Zap is meant to be my version of Pytorch Lightning, with significantly less complexity/functionality.

## Use
Here is a basic use case:

```python
from typing import Dict, Callable
from torch.utils.data import DataLoader
from zap import Trainer

model = ...

optimizer = ...

train_dataloader = ...
val_dataloader = ...

val_metrics : Dict[str, Callable] = ...

trainer = Trainer(
   model = model,
   train_dataloader = train_dataloader,
   val_dataloader = val_dataloader,
   val_metrics = val_metrics,
   optimizer = optimizer,
)

trainer.fit(num_epochs)
