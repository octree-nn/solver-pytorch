import datetime
import re
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def install_torch_stub():
  class _DecoratorContext:
    def __call__(self, fn):
      return fn

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc, tb):
      return False

  class _FakeTensor:
    is_cuda = False

    def detach(self):
      return self

    def item(self):
      return 0

    def numel(self):
      return 0

  class _FakeIndices:
    def __init__(self, values):
      self.values = list(values)

    def tolist(self):
      return list(self.values)

  class _Optimizer:
    def __init__(self, *args, **kwargs):
      pass

    def state_dict(self):
      return {}

    def load_state_dict(self, state):
      return None

    def zero_grad(self, *args, **kwargs):
      return None

    def step(self):
      return None

  class _Scheduler:
    def __init__(self, *args, **kwargs):
      pass

    def state_dict(self):
      return {}

    def load_state_dict(self, state):
      return None

    def step(self):
      return None

    def get_last_lr(self):
      return [0.0]

  class _GradScaler:
    def __init__(self, *args, **kwargs):
      pass

    def scale(self, value):
      return value

    def unscale_(self, optimizer):
      return None

    def step(self, optimizer):
      return None

    def update(self):
      return None

    def state_dict(self):
      return {}

    def load_state_dict(self, state):
      return None

  class _Module:
    def cuda(self, device=None):
      return self

    def parameters(self):
      return []

    def state_dict(self):
      return {}

    def load_state_dict(self, state):
      return None

  class _DistributedDataParallel(_Module):
    def __init__(self, module=None, **kwargs):
      self.module = module

    def parameters(self):
      if self.module is None:
        return []
      return self.module.parameters()

  class _Dataset:
    pass

  class _Sampler:
    pass

  class _DistributedSampler:
    def __init__(self, dataset, shuffle=True):
      self.dataset = dataset
      self.shuffle = shuffle

    def __iter__(self):
      return iter(range(len(self.dataset)))

    def set_epoch(self, epoch):
      return None

  class _DataLoader:
    def __init__(self, dataset, batch_size=1, num_workers=0, sampler=None,
                 collate_fn=None, pin_memory=False):
      self.dataset = dataset
      self.sampler = sampler
      self.collate_fn = collate_fn or (lambda x: x)

    def __iter__(self):
      return iter(())

    def __len__(self):
      return len(self.dataset)

  class _SummaryWriter:
    def __init__(self, *args, **kwargs):
      pass

    def add_scalar(self, *args, **kwargs):
      return None

  class _ProfileResult:
    def table(self, **kwargs):
      return ''

  class _Profile:
    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc, tb):
      return False

    def step(self):
      return None

    def key_averages(self, **kwargs):
      return _ProfileResult()

  torch = ModuleType('torch')
  torch.__version__ = '0.0'
  torch.Tensor = _FakeTensor
  torch.GradScaler = _GradScaler
  torch.from_numpy = lambda value: _FakeTensor()
  torch.randperm = lambda num: _FakeIndices(range(num))
  torch.arange = lambda num: _FakeIndices(range(num))
  torch.ones_like = lambda tensor: tensor
  torch.stack = lambda tensors, dim=0: _FakeTensor()
  torch.mean = lambda tensor: tensor
  torch.manual_seed = lambda seed: None
  torch.save = lambda *args, **kwargs: None
  torch.load = lambda *args, **kwargs: {}
  torch.no_grad = lambda: _DecoratorContext()
  torch.autocast = lambda *args, **kwargs: _DecoratorContext()

  torch.cuda = ModuleType('torch.cuda')
  torch.cuda.current_device = lambda: 0
  torch.cuda.synchronize = lambda: None
  torch.cuda.is_available = lambda: False
  torch.cuda.empty_cache = lambda: None
  torch.cuda.memory_reserved = lambda: 0
  torch.cuda.manual_seed = lambda seed: None
  torch.cuda.manual_seed_all = lambda seed: None
  torch.cuda.set_device = lambda device: None
  torch.cuda.amp = ModuleType('torch.cuda.amp')
  torch.cuda.amp.GradScaler = _GradScaler

  torch.backends = ModuleType('torch.backends')
  torch.backends.cudnn = ModuleType('torch.backends.cudnn')
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = False

  torch.nn = ModuleType('torch.nn')
  torch.nn.Module = _Module
  torch.nn.SyncBatchNorm = ModuleType('torch.nn.SyncBatchNorm')
  torch.nn.SyncBatchNorm.convert_sync_batchnorm = staticmethod(
      lambda model: model)
  torch.nn.parallel = ModuleType('torch.nn.parallel')
  torch.nn.parallel.DistributedDataParallel = _DistributedDataParallel
  torch.nn.utils = ModuleType('torch.nn.utils')
  torch.nn.utils.clip_grad_norm_ = lambda *args, **kwargs: None

  torch.optim = ModuleType('torch.optim')
  torch.optim.Optimizer = _Optimizer
  torch.optim.SGD = _Optimizer
  torch.optim.Adam = _Optimizer
  torch.optim.AdamW = _Optimizer
  torch.optim.lr_scheduler = ModuleType('torch.optim.lr_scheduler')
  torch.optim.lr_scheduler._LRScheduler = _Scheduler
  torch.optim.lr_scheduler.MultiStepLR = _Scheduler
  torch.optim.lr_scheduler.CosineAnnealingLR = _Scheduler
  torch.optim.lr_scheduler.LambdaLR = _Scheduler

  torch.distributed = ModuleType('torch.distributed')
  torch.distributed.get_world_size = lambda: 1
  torch.distributed.all_gather = lambda *args, **kwargs: None
  torch.distributed.init_process_group = lambda *args, **kwargs: None
  torch.distributed.barrier = lambda: None
  torch.distributed.is_initialized = lambda: False
  torch.distributed.destroy_process_group = lambda: None

  torch.multiprocessing = ModuleType('torch.multiprocessing')
  torch.multiprocessing.spawn = lambda *args, **kwargs: None

  torch.utils = ModuleType('torch.utils')
  torch.utils.data = ModuleType('torch.utils.data')
  torch.utils.data.Dataset = _Dataset
  torch.utils.data.Sampler = _Sampler
  torch.utils.data.DistributedSampler = _DistributedSampler
  torch.utils.data.DataLoader = _DataLoader
  torch.utils.tensorboard = ModuleType('torch.utils.tensorboard')
  torch.utils.tensorboard.SummaryWriter = _SummaryWriter

  torch.profiler = ModuleType('torch.profiler')
  torch.profiler.schedule = lambda *args, **kwargs: None
  torch.profiler.tensorboard_trace_handler = lambda *args, **kwargs: None
  torch.profiler.profile = lambda *args, **kwargs: _Profile()
  torch.profiler.ProfilerActivity = ModuleType('torch.profiler.ProfilerActivity')
  torch.profiler.ProfilerActivity.CPU = 'cpu'
  torch.profiler.ProfilerActivity.CUDA = 'cuda'

  modules = {
      'torch': torch,
      'torch.backends': torch.backends,
      'torch.backends.cudnn': torch.backends.cudnn,
      'torch.cuda': torch.cuda,
      'torch.cuda.amp': torch.cuda.amp,
      'torch.distributed': torch.distributed,
      'torch.multiprocessing': torch.multiprocessing,
      'torch.nn': torch.nn,
      'torch.nn.parallel': torch.nn.parallel,
      'torch.nn.utils': torch.nn.utils,
      'torch.optim': torch.optim,
      'torch.optim.lr_scheduler': torch.optim.lr_scheduler,
      'torch.profiler': torch.profiler,
      'torch.utils': torch.utils,
      'torch.utils.data': torch.utils.data,
      'torch.utils.tensorboard': torch.utils.tensorboard,
  }
  sys.modules.update(modules)


try:
  import torch  # noqa: F401
  import torch.nn  # noqa: F401
  import torch.utils.data  # noqa: F401
  import torch.utils.tensorboard  # noqa: F401
except Exception:
  stale_modules = [name for name in sys.modules if name == 'torch' or
                   name.startswith('torch.')]
  for name in stale_modules:
    sys.modules.pop(name, None)
  install_torch_stub()


def get_release() -> str:
  try:
    return metadata.version('thsolver')
  except metadata.PackageNotFoundError:
    setup_py = ROOT / 'setup.py'
    match = re.search(
        r"__version__\s*=\s*['\"]([^'\"]+)['\"]",
        setup_py.read_text(encoding='utf-8'))
    return match.group(1) if match else 'unknown'


project = 'solver-pytorch'
author = 'Peng-Shuai Wang'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)
release = get_release()
version = release


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

autosummary_generate = True
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

autodoc_member_order = 'bysource'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'torch': ('https://docs.pytorch.org/docs/stable/', None),
}

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 2,
}

epub_show_urls = 'footnote'
add_module_names = False


def setup(app):
  def skip(app, what, name, obj, skip, options):
    members = {
        '__dict__',
        '__init__',
        '__module__',
        '__repr__',
        '__weakref__',
    }
    return True if name in members else skip

  app.connect('autodoc-skip-member', skip)
