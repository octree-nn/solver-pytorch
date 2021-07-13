import os
import torch
import torch.nn
import torch.optim
import torch.distributed
import torch.multiprocessing
import torch.utils.data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .sampler import InfSampler, DistributedInfSampler


class AverageTracker:
  def __init__(self):
    self.value = None
    self.num = 0.0
    self.max_len = 70

  def update(self, value):
    if not value:
      return    # empty input, return

    value = {key: val.detach() for key, val in value.items()}
    if self.value is None:
      self.value = value
    else:
      for key, val in value.items():
        self.value[key] += val
    self.num += 1

  def average(self):
    return {key: val.item()/self.num for key, val in self.value.items()}

  @torch.no_grad()
  def average_all_gather(self):
    for key, tensor in self.value.items():
      tensors_gather = [torch.ones_like(tensor)
                        for _ in range(torch.distributed.get_world_size())]
      torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
      tensors = torch.stack(tensors_gather, dim=0)
      self.value[key] = torch.mean(tensors)

  def log(self, epoch, writer=None):
    avg = self.average()
    msg = 'Epoch: %d; ' % epoch
    for key, val in avg.items():
      msg += '%s: %.3f; ' % (key, val)
      if writer:
        writer.add_scalar(key, val, epoch)
    if len(msg) > self.max_len:
      msg = msg[:self.max_len] + ' ...'
    tqdm.write(msg)


class Solver:
  def __init__(self, FLAGS, is_master=True):
    self.FLAGS = FLAGS
    self.is_master = is_master
    self.world_size = len(FLAGS.SOLVER.gpu)
    self.device = torch.cuda.current_device()
    self.writer = None  # SummaryWriter

  def get_model(self):
    raise NotImplementedError

  def get_dataset(self, flags, training):
    raise NotImplementedError

  def train_step(self, batch):
    raise NotImplementedError

  def test_step(self, batch):
    raise NotImplementedError

  def result_callback(self, avg_tracker: AverageTracker, epoch):
    pass  # additional operations based on the avg_tracker

  def config_dataloader(self, training=True):
    if training:
      self.train_loader = self.get_dataloader(training=True)
      self.train_iter = iter(self.train_loader)
    self.test_loader = self.get_dataloader(training=False)
    self.test_iter = iter(self.test_loader)

  def get_dataloader(self, training):
    flags = self.FLAGS.DATA.train if training else self.FLAGS.DATA.test
    dataset, collate_fn = self.get_dataset(flags)

    if self.world_size > 1:
      sampler = DistributedInfSampler(dataset, shuffle=training)
    else:
      sampler = InfSampler(dataset, shuffle=training)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=flags.batch_size, num_workers=flags.num_workers,
        sampler=sampler, collate_fn=collate_fn, pin_memory=True, shuffle=False)
    return data_loader

  def config_model(self):
    model = self.get_model(self.FLAGS.MODEL)
    # if self.world_size > 1:
    #  model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(device=self.device)
    if self.world_size > 1:
      model = torch.nn.parallel.DistributedDataParallel(
          module=model, device_ids=[self.device],
          output_device=self.device, broadcast_buffers=False,
          find_unused_parameters=False)
    if self.is_master:
      print(model)
    self.model = model

  def configure_optimizer(self):
    # The learning rate scales with regard to the world_size
    lr = self.FLAGS.SOLVER.lr * self.world_size
    self.optimizer = torch.optim.SGD(
        self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    if self.FLAGS.SOLVER.lr_type == 'step':
      self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
          self.optimizer, milestones=self.FLAGS.SOLVER.step_size, gamma=0.1)
    elif self.FLAGS.SOLVER.lr_type == 'cos':
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          self.optimizer, self.FLAGS.SOLVER.max_epoch, eta_min=0.001)
    else:
      raise ValueError

  def configure_log(self):
    if not self.is_master:
      return

    self.logdir = self.FLAGS.SOLVER.logdir
    self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
    print('Logdir: ' + self.logdir)
    self.writer = SummaryWriter(self.logdir, flush_secs=20)
    if not os.path.exists(self.ckpt_dir):
      os.makedirs(self.ckpt_dir)

  def train_epoch(self, epoch):
    self.model.train()
    if self.world_size > 1:
      self.train_loader.sampler.set_epoch(epoch)

    train_tracker = AverageTracker()
    for _ in tqdm(range(len(self.train_loader)), ncols=80, leave=False):
      self.optimizer.zero_grad()

      # forward
      batch = self.train_iter.next()
      output = self.train_step(batch)

      # backward
      output['train/loss'].backward()
      self.optimizer.step()

      # track the averaged tensors
      train_tracker.update(output)

    # save logs
    if self.world_size > 1:
      train_tracker.average_all_gather()
    if self.is_master:
      train_tracker.log(epoch, self.writer)

  def test_epoch(self, epoch):
    self.model.eval()
    test_tracker = AverageTracker()
    for _ in tqdm(range(len(self.test_loader)), ncols=80, leave=False):
      batch = self.test_iter.next()
      with torch.no_grad():
        output = self.test_step(batch)
      test_tracker.update(output)
    if self.world_size > 1:
      test_tracker.average_all_gather()
    if self.is_master:
      test_tracker.log(epoch, self.writer)
      self.result_callback(test_tracker, epoch)

  def dump_checkpoint(self, epoch):
    if not self.is_master:
      return

    # clean up
    ckpts = sorted(os.listdir(self.ckpt_dir))
    ckpts = [ckpt for ckpt in ckpts if ckpt.endswith('.pth')]
    if len(ckpts) > self.FLAGS.SOLVER.ckpt_num:
      for ckpt in ckpts[:-self.FLAGS.SOLVER.ckpt_num]:
        os.remove(os.path.join(self.ckpt_dir, ckpt))

    # dump ckpt
    ckpt_name = os.path.join(self.ckpt_dir, 'model_%05d.pth' % epoch)
    torch.save(self.model.state_dict(), ckpt_name)

  def train(self):
    self.config_model()
    self.config_dataloader()
    self.configure_optimizer()
    self.configure_log()

    for epoch in tqdm(range(1, self.FLAGS.SOLVER.max_epoch+1), ncols=80):
      # training epoch
      self.train_epoch(epoch)

      # update learning rate
      self.scheduler.step()
      if self.is_master:
        lr = self.scheduler.get_last_lr()  # lr is a list
        self.writer.add_scalar('train/lr', lr[0], epoch)

      # testing or not
      if epoch % self.FLAGS.SOLVER.test_every_epoch != 0:
        continue

      # testing epoch
      self.test_epoch(epoch)

      # checkpoint
      self.dump_checkpoint(epoch)

  def test(self):
    self.config_model()
    self.config_dataloader(training=False)
    trained_dict = torch.load(self.FLAGS.SOLVER.ckpt)  # TODO: check multi-gpu
    self.model.load_state_dict(trained_dict)
    self.test_epoch(epoch=0)

  def profile(self):
    ''' Set `DATA.train.num_workers 0` when using this function'''
    self.config_model()
    self.config_dataloader()

    # warm up
    batch = next(iter(self.train_loader))
    for _ in range(3):
      output = self.train_step(batch)
      output['train/loss'].backward()

    # profile
    with torch.autograd.profiler.profile(
            use_cuda=True, profile_memory=True,
            with_stack=True, record_shapes=True) as prof:
      output = self.train_step(batch)
      output['train/loss'].backward()

    json = os.path.join(self.FLAGS.SOLVER.logdir, 'trace.json')
    print('Save the profile into: ' + json)
    prof.export_chrome_trace(json)
    print(prof.key_averages(group_by_stack_n=10)
              .table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_stack_n=10)
              .table(sort_by="cuda_memory_usage", row_limit=10))

  def run(self):
    eval('self.%s()' % self.FLAGS.SOLVER.run)

  @staticmethod
  def main_worker(gpu, FLAGS, TheSolver):
    world_size = len(FLAGS.SOLVER.gpu)
    if world_size > 1:
      # Set the GPU to use.
      torch.cuda.set_device(gpu)
      # Initialize the process group. Currently, the code only supports the
      # `single node + multiple GPU` mode, so the rank is equal to gpu id.
      torch.distributed.init_process_group(
          backend='nccl', init_method=FLAGS.SOLVER.dist_url,
          world_size=world_size, rank=gpu)
      # Master process is responsible for logging, writing and loading
      # checkpoints. In the multi GPU setting, we assign the master role to the
      # rank 0 process.
      is_master = gpu == 0
      solver = TheSolver(FLAGS, is_master)
    else:
      solver = TheSolver(FLAGS, is_master=True)
    solver.run()

  @staticmethod
  def main(FLAGS, TheSolver):
    num_gpus = len(FLAGS.SOLVER.gpu)
    if num_gpus > 1:
      torch.multiprocessing.spawn(
          Solver.main_worker, nprocs=num_gpus, args=(FLAGS, TheSolver))
    else:
      Solver.main_worker(0, FLAGS, TheSolver)
