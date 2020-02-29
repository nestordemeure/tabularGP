# Tabular GP

The aim of this repository is to make it easy to use gaussian process on tabular data with an implementation built on top of the [fastai V1 framework](https://docs.fast.ai/).

If you are a gaussian process expert, you might be better served by [gpytorch](https://gpytorch.ai/) as our focus is on accesibility and out-of-the-box experiences rather than exhaustivity and flexibility.

**WARNING: this is a work in progress that is still very incomplete (see our TODO list).**

## Capabilities

- regression on one or more targets
- classification (TODO)
- uncertainty on the outputs
- feature importance estimation (TODO)
- transfer-learning to recycle models (TODO)

## TODO

### Documentation

- add usage example to readme
- add demo notebook

### Kernel

- add neural network encoder kernel
- make abstract classes for the various kinds of kernel
- add matern kernel
- add a date/time specific kernel
- add possibility to pass a list of kernels to the tabular kernels (to have one specific kernel per column)

### Model

- add classification
- add feature importance for kernels that supports it
- add transfer learning
- use kmean clustering to find representative inducing points

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*