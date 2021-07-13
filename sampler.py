import torch
from torch.utils.data import Sampler, DistributedSampler, Dataset
from typing import Optional


class InfSampler(Sampler[int]):
  def __init__(self, dataset, shuffle=True):
    self.dataset = dataset
    self.shuffle = shuffle
    self.reset_sampler()

  def reset_sampler(self):
    num = len(self.dataset)
    indices = torch.randperm(num) if self.shuffle else torch.arange(num)
    self.indices = indices.tolist()
    self.iter_num = 0

  def __iter__(self):
    return self

  def __next__(self):
    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value

  def __len__(self):
    return len(self.dataset)


class DistributedInfSampler(DistributedSampler):
  def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
               rank: Optional[int] = None, shuffle: bool = True,
               seed: int = 0, drop_last: bool = False) -> None:
    super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
    self.reset_sampler()

  def reset_sampler(self):
    self.indices = list(super().__iter__())
    self.iter_num = 0

  def __iter__(self):
    return self

  def __next__(self):
    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value
