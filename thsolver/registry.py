# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

_model_entrypoints = {}
_dataset_entrypoints = {}


def register_model(fn):
  r''' Registers a model factory by its function name.

  Args:
    fn (callable): The model factory to register.

  Returns:
    callable: The input factory, which keeps decorator usage convenient.
  '''

  name = fn.__name__
  _model_entrypoints[name] = fn
  return fn


def model_entrypoints(name):
  r''' Returns the registered model factory with the given name.

  Args:
    name (str): The model name.
  '''

  return _model_entrypoints[name]


def is_model(name):
  r''' Checks whether a model factory has been registered.

  Args:
    name (str): The model name.
  '''

  return name in _model_entrypoints


def list_models():
  r''' Returns all registered model names. '''

  return list(_model_entrypoints.keys())


def build_model(config, **kwargs):
  r''' Builds a registered model from a config node.

  Args:
    config: A config node containing the field ``name``.
    **kwargs: Additional arguments forwarded to the model factory.

  Returns:
    object: The model created by the registered factory.
  '''

  name = config.name.lower()
  if not is_model(name):
    raise ValueError(f'Unkown model: {name}')
  return model_entrypoints(name)(config, **kwargs)


def register_dataset(fn):
  r''' Registers a dataset factory by its function name.

  Args:
    fn (callable): The dataset factory to register.

  Returns:
    callable: The input factory, which keeps decorator usage convenient.
  '''

  name = fn.__name__
  _dataset_entrypoints[name] = fn
  return fn


def dataset_entrypoints(name):
  r''' Returns the registered dataset factory with the given name.

  Args:
    name (str): The dataset name.
  '''

  return _dataset_entrypoints[name]


def is_dataset(name):
  r''' Checks whether a dataset factory has been registered.

  Args:
    name (str): The dataset name.
  '''

  return name in _dataset_entrypoints


def build_dataset(config, **kwargs):
  r''' Builds a registered dataset from a config node.

  Args:
    config: A config node containing the field ``name``.
    **kwargs: Additional arguments forwarded to the dataset factory.

  Returns:
    object: The dataset tuple or object created by the registered factory.
  '''

  name = config.name.lower()
  if not is_dataset(name):
    raise ValueError(f'Unkown dataset: {name}')
  return dataset_entrypoints(name)(config, **kwargs)


def list_datasets():
  r''' Returns all registered dataset names. '''

  return list(_dataset_entrypoints.keys())
