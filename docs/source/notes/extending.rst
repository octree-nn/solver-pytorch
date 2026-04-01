Extending thsolver
==================

The project is designed to stay small and extensible. Most customization
happens through factories and by overriding a handful of methods.


Model and Dataset Registries
----------------------------

The registry helpers in :mod:`thsolver.registry` let you decouple experiment
configuration from Python imports.

.. code-block:: python

   from thsolver import register_dataset, register_model


   @register_model
   def my_model(flags):
       return MyModel(flags)


   @register_dataset
   def my_dataset(flags):
       dataset = MyDataset(flags.location)
       return dataset, my_collate_fn

When ``MODEL.name`` is ``my_model``, :func:`thsolver.registry.build_model`
returns ``my_model(flags)``. The same pattern applies to datasets.

.. note::

   ``build_model`` and ``build_dataset`` call ``config.name.lower()`` before the
   lookup. In practice, that means registered factory names should be lowercase
   to avoid mismatches.


Custom Result Handling
----------------------

Override :meth:`thsolver.solver.Solver.result_callback` when you need additional
logic after a test epoch finishes. Typical uses include:

- writing task-specific summaries
- computing a metric that depends on all batch outputs
- synchronizing with an external evaluator

The callback receives the epoch-level :class:`thsolver.tracker.AverageTracker`.


Custom Evaluation Jobs
----------------------

``evaluate`` mode is intentionally separate from ``test`` mode. Use it when the
primary output is not an averaged scalar metric, for example:

- exporting predictions
- computing mesh or point-cloud reconstructions
- saving visualizations
- running validation code that writes files to disk

Implement the side effects inside :meth:`thsolver.solver.Solver.eval_step`.


Replacing the Dataset Helper
----------------------------

You do not have to use :class:`thsolver.dataset.Dataset`. Any
``torch.utils.data.Dataset`` can be returned from
:meth:`thsolver.solver.Solver.get_dataset`. The only requirement is that your
``collate_fn`` and step methods agree on the batch format.
