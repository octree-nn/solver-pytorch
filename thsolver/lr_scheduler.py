# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import math
from bisect import bisect_right
import torch.optim.lr_scheduler as LR


def multi_step(optimizer, flags):
  r''' Builds a multi-step learning-rate scheduler.

  Args:
    optimizer: The optimizer to schedule.
    flags: The solver config node.
  '''

  return LR.MultiStepLR(optimizer, flags.milestones, flags.gamma)


def cos(optimizer, flags):
  r''' Builds a cosine annealing learning-rate scheduler.

  Args:
    optimizer: The optimizer to schedule.
    flags: The solver config node.
  '''

  return LR.CosineAnnealingLR(optimizer, flags.max_epoch, eta_min=flags.lr_min)


def poly(optimizer, flags):
  r''' Builds a polynomial-decay learning-rate scheduler.

  Args:
    optimizer: The optimizer to schedule.
    flags: The solver config node.
  '''

  def lr_lambda(epoch):
    r''' Returns the polynomial decay ratio for the current epoch. '''
    return (1 - epoch / flags.max_epoch) ** flags.lr_power

  return LR.LambdaLR(optimizer, lr_lambda)


def constant(optimizer, flags):
  r''' Builds a constant learning-rate scheduler.

  Args:
    optimizer: The optimizer to schedule.
    flags: The solver config node.
  '''

  def lr_lambda(epoch):
    r''' Returns the constant learning-rate ratio. '''
    return 1

  return LR.LambdaLR(optimizer, lr_lambda)


def cos_warmup(optimizer, flags):
  r''' Builds a cosine schedule with linear warmup.

  Args:
    optimizer: The optimizer to schedule.
    flags: The solver config node.
  '''

  def lr_lambda(epoch):
    r''' Returns the warmup or cosine-decay ratio for the current epoch. '''

    warmup = flags.warmup_epoch
    warmup_init = flags.warmup_init
    if epoch <= warmup:
      return (1 - warmup_init) * epoch / warmup + warmup_init
    else:
      lr_min = flags.lr_min
      ratio = (epoch - warmup) / (flags.max_epoch - warmup)
      return lr_min + 0.5 * (1.0 - lr_min) * (1 + math.cos(math.pi * ratio))

  return LR.LambdaLR(optimizer, lr_lambda)


def poly_warmup(optimizer, flags):
  r''' Builds a polynomial schedule with linear warmup.

  Args:
    optimizer: The optimizer to schedule.
    flags: The solver config node.
  '''

  def lr_lambda(epoch):
    r''' Returns the warmup or polynomial-decay ratio for the current epoch. '''

    warmup = flags.warmup_epoch
    warmup_init = flags.warmup_init
    if epoch <= warmup:
      return (1 - warmup_init) * epoch / warmup + warmup_init
    else:
      ratio = (epoch - warmup) / (flags.max_epoch - warmup)
      return (1 - ratio) ** flags.lr_power

  return LR.LambdaLR(optimizer, lr_lambda)


def step_warmup(optimizer, flags):
  r''' Builds a step schedule with linear warmup.

  Args:
    optimizer: The optimizer to schedule.
    flags: The solver config node.
  '''

  def lr_lambda(epoch):
    r''' Returns the warmup or step-decay ratio for the current epoch. '''

    warmup = flags.warmup_epoch
    warmup_init = flags.warmup_init
    if epoch <= warmup:
      return (1 - warmup_init) * epoch / warmup + warmup_init
    else:
      milestones = sorted(flags.milestones)
      return flags.gamma ** bisect_right(milestones, epoch)

  return LR.LambdaLR(optimizer, lr_lambda)


def get_lr_scheduler(optimizer, flags):
  r''' Builds the configured learning-rate scheduler.

  Args:
    optimizer: The optimizer to schedule.
    flags: The solver config node containing ``lr_type``.

  Returns:
    torch.optim.lr_scheduler._LRScheduler: The configured scheduler.
  '''

  lr_dict = {'cos': cos, 'step': multi_step, 'cos': cos, 'poly': poly,
             'constant': constant, 'cos_warmup': cos_warmup,
             'poly_warmup': poly_warmup, 'step_warmup': step_warmup}
  lr_func = lr_dict[flags.lr_type]
  return lr_func(optimizer, flags)
