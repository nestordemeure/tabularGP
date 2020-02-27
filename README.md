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

- i got approximate gp working but it would be nice to get exact gp to scale
- the code uses a numbers of hacks to run without numerical problem / gpytorch bugs, we might be able to improve on that
- i need proper kernels for the input instead of the current placeholder

- validate tabularGPexact on multitask regression
- reproduce indexkernel bug without fastai and send it to gpytorch developpers (or find fix)
- use resilient cholesky as template for the noise addition
- implement universal combinator and use it to get a product of sums kernel
  - compare it with a dnn kernel
  - and a sum kernel
  - and a product kernel
- use kmediod clustering to select inducing points (or just closest points to kmeans to get faster)

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*