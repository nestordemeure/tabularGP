# Tabular GP

**WARNING: this is a work in progress that is still subject to deep modifications.**

The aim of this repository is to make it easy to use gaussian process on tabular data within the [fastai V1 framework](https://docs.fast.ai/), experiment with various kernel types and evaluate their efficiency in this domain.

## TODO

### Documentation

- add usage example
- add demo notebook

### Kernel

- add possibility to deactivate scaling on kernels
- add neural network encoder kernel
- add Weighted Product kernel
- add universal combinator kernel

### Model

- add methods to do classification
- add feature importance for kernels that support it out of the box
- add transfer learning (by reusing the kernel)
- use kmean clustering to find representative inducing points

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*