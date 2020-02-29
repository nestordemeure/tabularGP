# Tabular GP

**WARNING: this is a work in progress that is still subject to deep modifications.**

The aim of this repository is to make it easy to use gaussian process on tabular data within the [fastai V1 framework](https://docs.fast.ai/), experiment with various kernel types and evaluate their efficiency in this domain.

## Capabilities

- regression
- classification (TODO)
- provides a well calibrated uncertainty on the outputs
- compute feature importance (TODO)
- transfer learning (TODO)

## TODO

### Documentation

- add usage example
- add demo notebook

### Kernel

- add neural network encoder kernel
- add Weighted Product kernel

### Model

- add methods to do classification
- add feature importance for kernels that support it out of the box
- add transfer learning (by reusing the kernel)
- use kmean clustering to find representative inducing points

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*