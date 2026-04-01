Distributed Training
====================

``thsolver`` supports two distributed launch styles controlled by
``SOLVER.ddp_mode``.


Spawn Mode
----------

``spawn`` is the default. It is best suited to single-node, multi-GPU training.

- ``SOLVER.gpu`` defines the GPU ids to use.
- ``SOLVER.port`` defines the localhost NCCL rendezvous port.
- The solver launches one worker per listed GPU with
  :func:`torch.multiprocessing.spawn`.

When only one GPU id is provided, the same code path still works and runs a
single worker.


torchrun Mode
-------------

``torchrun`` uses the environment variables created by
``torch.distributed.run``. This is the better fit for multi-node launches or
when you already standardize on ``torchrun``.

.. code-block:: none

   torchrun --nproc_per_node=4 train.py --config configs/experiment.yaml

In this mode, ``WORLD_SIZE``, ``RANK``, and ``LOCAL_RANK`` come from the launch
environment rather than from ``SOLVER.gpu``.


Master Process Responsibilities
-------------------------------

Only rank 0 performs:

- TensorBoard and CSV logging
- checkpoint save and restore bookkeeping
- best-checkpoint updates
- console logging intended for the user

All other ranks participate in forward and backward passes and synchronize
epoch-level metrics through :meth:`thsolver.tracker.AverageTracker.average_all_gather`.


Reproducibility
---------------

Set ``SOLVER.rand_seed`` to a positive integer to enable deterministic seeding
for Python, NumPy, and PyTorch. When a seed is fixed, the solver also disables
``cudnn.benchmark`` and enables deterministic CuDNN behavior.
