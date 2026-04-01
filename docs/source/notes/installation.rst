Installation
============

Requirements
------------

``thsolver`` depends on:

- `PyTorch <https://pytorch.org/get-started/locally/>`_
- ``numpy``
- ``yacs``
- ``tqdm``
- ``packaging``

If you want TensorBoard logging, install ``tensorboard`` as well. The package
imports :class:`torch.utils.tensorboard.SummaryWriter`, so a standard PyTorch
training environment will usually already include it.


Installation via Pip
--------------------

Install the published package after PyTorch is available:

.. code-block:: none

   pip install thsolver


Installation from Source
------------------------

Clone the repository and install it into the current Python environment:

.. code-block:: none

   git clone https://github.com/octree-nn/solver-pytorch.git
   pip install ./solver-pytorch


Documentation Dependencies
--------------------------

To build the docs locally, install the Sphinx requirements:

.. code-block:: none

   pip install -r docs/requirements.txt

Then build the HTML output from the repository root:

.. code-block:: none

   cd docs
   make html

On Windows, use:

.. code-block:: none

   cd docs
   make.bat html
