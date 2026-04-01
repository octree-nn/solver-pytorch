Configuration
=============

``thsolver`` uses :class:`yacs.config.CfgNode` to define and merge experiment
configuration. The default tree lives in :mod:`thsolver.config`.


How Configuration Is Loaded
---------------------------

The command-line parser in :func:`thsolver.config.parse_args` supports:

- ``--config path/to/file.yaml`` to load a configuration file
- a ``BASE`` key inside configuration files for hierarchical composition
- trailing key-value overrides such as ``SOLVER.max_epoch 10``

The parsed configuration is frozen before training starts, and the final merged
config is written to ``<logdir>/all_configs.yaml`` for reproducibility.


Top-Level Sections
------------------

The built-in tree contains these main sections:

- ``SOLVER`` for runtime, optimization, checkpointing, logging, and DDP
- ``DATA.train`` and ``DATA.test`` for dataset and dataloader settings
- ``MODEL`` for model construction options
- ``LOSS`` for task-specific loss configuration
- ``SYS`` for bookkeeping such as the captured command line


Important SOLVER Fields
-----------------------

Some of the most frequently used options are:

- ``SOLVER.run``: ``train``, ``test``, ``evaluate``, or ``profile``
- ``SOLVER.logdir``: directory used for logs, checkpoints, and config backups
- ``SOLVER.alias``: optional suffix added to ``logdir``; the word ``time`` is
  replaced with the current timestamp
- ``SOLVER.ckpt``: checkpoint path to restore explicitly
- ``SOLVER.type``: optimizer name, currently ``sgd``, ``adam``, or ``adamw``
- ``SOLVER.lr_type``: scheduler strategy such as ``step``, ``cos``,
  ``poly``, ``constant``, or warmup variants
- ``SOLVER.best_val``: metric selector such as ``min:loss`` or ``max:accu``
- ``SOLVER.amp_mode``: ``none``, ``fp16``, or ``bf16``
- ``SOLVER.ddp_mode``: ``spawn`` or ``torchrun``


Dataset Configuration
---------------------

``DATA.train`` and ``DATA.test`` share the same structure. Useful fields include:

- ``name``: dataset factory name used by :func:`thsolver.registry.build_dataset`
- ``location``: dataset root directory
- ``filelist``: text file describing samples
- ``batch_size`` and ``num_workers``: dataloader settings
- ``shuffle``: whether the infinite sampler should randomize sample order
- ``take``: limit the number of examples for quick experiments
- ``disable``: skip the train or test dataloader entirely

The built-in :class:`thsolver.dataset.Dataset` expects a filelist where each
line contains a relative filename and an optional integer label.


Example Override
----------------

.. code-block:: none

   python train.py --config configs/experiment.yaml \
       SOLVER.alias debug \
       SOLVER.max_epoch 5 \
       DATA.train.take 64
