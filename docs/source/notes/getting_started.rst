Getting Started
===============

The normal workflow is:

1. Define a model factory and optionally register it with
   :func:`thsolver.registry.register_model`.
2. Define a dataset factory and optionally register it with
   :func:`thsolver.registry.register_dataset`.
3. Subclass :class:`thsolver.solver.Solver`.
4. Implement the hooks for your task.
5. Launch the run with :meth:`thsolver.solver.Solver.main`.


Minimal Solver Skeleton
-----------------------

The base solver owns the training loop. Your subclass supplies the pieces that
are task-specific.

.. code-block:: python

   import torch
   from thsolver import Solver, build_dataset, build_model


   class DemoSolver(Solver):

       def get_model(self, flags):
           return build_model(flags)

       def get_dataset(self, flags):
           return build_dataset(flags)

       def train_step(self, batch):
           output = self.model(batch)
           loss = output["loss"]
           return {
               "train/loss": loss,
               "train/metric": output["metric"],
           }

       @torch.no_grad()
       def test_step(self, batch):
           output = self.model(batch)
           return {
               "test/loss": output["loss"],
               "test/metric": output["metric"],
           }

       def eval_step(self, batch):
           output = self.model(batch)
           # write predictions or accumulate task-specific results here


   if __name__ == "__main__":
       DemoSolver.main()

.. note::

   The example above is intentionally schematic. ``thsolver`` does not impose a
   fixed batch structure or model signature. Your dataloader decides what a
   batch looks like, and your hooks decide how to move it to CUDA, run the
   model, and compute losses or metrics.


Required Hooks
--------------

The following methods are expected in most projects:

- :meth:`thsolver.solver.Solver.get_model` returns a ``torch.nn.Module``.
- :meth:`thsolver.solver.Solver.get_dataset` returns ``(dataset, collate_fn)``.
- :meth:`thsolver.solver.Solver.train_step` must return a dict containing the
  key ``"train/loss"``.
- :meth:`thsolver.solver.Solver.test_step` returns test losses or metrics that
  should be averaged over the epoch.
- :meth:`thsolver.solver.Solver.eval_step` is used for evaluation-only jobs such
  as dumping predictions.


Run Modes
---------

``SOLVER.run`` selects which entry point the base class will execute:

- ``train`` runs the full training loop.
- ``test`` loads a checkpoint and evaluates on the test loader.
- ``evaluate`` calls :meth:`thsolver.solver.Solver.eval_step` for side-effect
  evaluation jobs.
- ``profile`` runs a short PyTorch profiler session on the training step.


What the Base Class Handles
---------------------------

Once the hooks are in place, :class:`thsolver.solver.Solver` takes care of:

- dataloader creation with infinite samplers
- optimizer and learning-rate scheduler setup
- mixed precision with ``none``, ``fp16``, or ``bf16``
- checkpoint save and restore
- TensorBoard logging and CSV logging
- best-checkpoint tracking through ``SOLVER.best_val``
- single-node spawn-based DDP and ``torchrun``-based launches
