import math
import torch.optim.lr_scheduler as LR


def multi_step(optimizer, flags):
  return LR.MultiStepLR(optimizer, milestones=flags.step_size, gamma=0.1)


def cos(optimizer, flags):
  return LR.CosineAnnealingLR(optimizer, flags.max_epoch, eta_min=flags.lr_min)


def poly(optimizer, flags):
  lr_lambda = lambda epoch: (1 - epoch / flags.max_epoch) ** flags.lr_power
  return LR.LambdaLR(optimizer, lr_lambda)


def constant(optimizer, flags):
  lr_lambda = lambda epoch: 1
  return LR.LambdaLR(optimizer, lr_lambda)


def cos_warmup(optimizer, flags):
  def lr_lambda(epoch):
    warmup = flags.warmp_epoch
    if epoch <= warmup:
      return epoch / warmup
    else:
      ratio = (epoch - warmup) / (flags.max_epoch - warmup)
      return 0.0001 + 0.49995 * (1 + math.cos(math.pi * ratio))
  return LR.LambdaLR(optimizer, lr_lambda)


def get_lr_scheduler(optimizer, flags):
  lr_dict = {'cos': cos, 'step': multi_step, 'cos': cos, 'poly': poly,
             'constant': constant, 'cos_warmup': cos_warmup}
  lr_func = lr_dict[flags.lr_type]
  return lr_func(optimizer, flags)
