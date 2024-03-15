import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedRandomSampler(Sampler):
    """
    DistributedRandomSampler randomly samples from the dataset WITH replacement.

    Args:
        num_samples (int) : the number of samples per GPU.
            e.g. if we want each GPU to ingest N batches of batch size B, then each process will deal with N*B samples.
    """

    def __init__(self, data_source, num_samples, seed = 0):
        self.data_source = data_source
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        # Deterministically shuffle based on seed
        g = torch.Generator()
        g.manual_seed(self.seed)

        indices = torch.randint(
            low = 0,
            high = len(self.data_source),
            size = (self.num_samples * self.num_replicas,),
            generator = g,
        ).tolist()

        # Assign each process a disjoint subset of these indices
        indices = indices[self.rank::self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples




