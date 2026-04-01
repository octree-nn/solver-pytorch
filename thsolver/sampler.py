# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from torch.utils.data import Sampler, DistributedSampler, Dataset


class InfSampler(Sampler):
  r''' An infinite sampler that cycles through dataset indices.

  Args:
    dataset (Dataset): The dataset to sample from.
    shuffle (bool): If True, shuffles the sample order between cycles.
  '''

  def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
    r''' Initializes the sampler. '''

    self.dataset = dataset
    self.shuffle = shuffle
    self.reset_sampler()

  def reset_sampler(self):
    r''' Resets the sampled index order for the next pass over the dataset. '''

    num = len(self.dataset)
    indices = torch.randperm(num) if self.shuffle else torch.arange(num)
    self.indices = indices.tolist()
    self.iter_num = 0

  def __iter__(self):
    r''' Returns the sampler iterator. '''

    return self

  def __next__(self):
    r''' Returns the next sample index and restarts when exhausted. '''

    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value

  def __len__(self):
    r''' Returns the dataset size used by the sampler. '''

    return len(self.dataset)


class DistributedInfSampler(DistributedSampler):
  r''' An infinite sampler for distributed data parallel training.

  Args:
    dataset (Dataset): The dataset to sample from.
    shuffle (bool): If True, shuffles the sample order between cycles.
  '''

  def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
    r''' Initializes the distributed sampler. '''

    super().__init__(dataset, shuffle=shuffle)
    self.reset_sampler()

  def reset_sampler(self):
    r''' Rebuilds the local rank indices from the base distributed sampler. '''

    self.indices = list(super().__iter__())
    self.iter_num = 0

  def __iter__(self):
    r''' Returns the sampler iterator. '''

    return self

  def __next__(self):
    r''' Returns the next rank-local sample index and restarts when exhausted. '''

    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value
