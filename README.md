# Tabular GP

**WARNING: this is a work in progress that is not usable yet.**

The aim of this repository is to make it easy to use gaussian process on tabular data within the fastai (V1) framework, 
experiment with various kernel types 
and evaluate their efficiency in this domain.

This repository builds on [gpytorch](https://gpytorch.ai/) and [fastai V1](https://docs.fast.ai/).

## Usage

Before using this project, you will need to install [fastai V1](https://docs.fast.ai/) and [gpytorch](https://gpytorch.ai/):

```
pip install fastai
pip install gpytorch
```

TODO add usage example

## TODO

- implement various kernels (deep learning encoder, classical kernels, all of those with various combinaisons methods)
- implement regression models with one output
- implement regression models with several outputs
- implement classification models

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*