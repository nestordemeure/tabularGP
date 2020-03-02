# Tabular GP

The aim of this repository is to make it easy to use gaussian process on tabular data with an implementation built on top of the [fastai V1 framework](https://docs.fast.ai/).

If you are a gaussian process expert, you might be better served by [gpytorch](https://gpytorch.ai/) as our focus is on accesibility and out-of-the-box experiences rather than exhaustivity and flexibility.

## Features

- regression on one or more targets
- classification
- uncertainty on the outputs
- feature importance estimation
- transfer-learning to recycle models (TODO)
- various kernels including a neural-network based kernel
- naturally resistant to overfitting

## Notes

A learning rate within  `[1e-2;1e-1]` seems like a good default.

The loss can sometimes decrease while the error is slightly increasing, this is due to the model fighting overfitting and improving its calibration.

Using SGD instead of Adam (`opt_func=optim.SGD`) is sometimes very beneficial with this kind of model but it can also lead to numerically unstable situations.

## TODO

#### Various

- add usage example to readme
- add demo notebook
- compare to neural network baselines

#### Kernel

- add a date/time specific kernel
- add possibility to pass a list of kernels to the tabular kernels (to have one specific kernel per column)

#### Model

- explore other likelihoods for classification (softmax)
- add transfer learning

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*