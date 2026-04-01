:github_url: https://github.com/octree-nn/solver-pytorch

solver-pytorch
==============

.. image:: https://static.pepy.tech/badge/thsolver
   :target: https://pepy.tech/project/thsolver
   :alt: Downloads

.. image:: https://img.shields.io/pypi/v/thsolver
   :target: https://pypi.org/project/thsolver/
   :alt: PyPI

``solver-pytorch`` provides the lightweight training infrastructure behind
``thsolver``. It wraps the repetitive parts of PyTorch experiments such as
configuration parsing, dataloader setup, checkpointing, logging, distributed
training, and evaluation, while leaving the model and batch logic in user code.

The project is intentionally small. Instead of introducing a large framework,
it gives you a base :class:`thsolver.solver.Solver`, a simple filelist-based
:class:`thsolver.dataset.Dataset`, registry helpers for models and datasets, and
utility modules for samplers, learning-rate schedules, and metric tracking.

Key benefits of ``thsolver`` include:

- **Simple experiment loops**. Subclass :class:`thsolver.solver.Solver` and
  implement only the hooks that define your task.
- **Config-driven runs**. Use :mod:`yacs` configuration files with command-line
  overrides and automatic config backups in the log directory.
- **Built-in training utilities**. Reuse checkpointing, TensorBoard logging,
  learning-rate scheduling, mixed precision, and DDP launch modes.
- **Easy extension points**. Register model and dataset factories with small
  decorators and keep the rest of the pipeline generic.

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/getting_started
   notes/configuration
   notes/extending
   notes/distributed

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   modules/config
   modules/solver
   modules/dataset
   modules/registry
   modules/sampler
   modules/lr_scheduler
   modules/tracker

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
