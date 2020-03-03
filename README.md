# Tabular GP

The aim of this repository is to make it easy to use gaussian process on tabular data with an implementation built on top of the [fastai V1 framework](https://docs.fast.ai/).

If you are a gaussian process expert, you might be better served by [gpytorch](https://gpytorch.ai/) as our focus is on accesibility and out-of-the-box experiences rather than exhaustivity and flexibility.

## Usage

```python
# train a learner on a classification task
learn = tabularGP_learner(data, nb_training_points=50, metrics=accuracy)
learn.fit_one_cycle(10, max_lr=1e-3)

# display the feature importance
glearn.plot_feature_importance()
```

## Features

- regression on one or more targets
- classification
- uncertainty on the outputs
- feature importance estimation
- transfer-learning to recycle models (TODO)
- various kernels including a neural-network based kernel
- naturally resistant to overfitting
- very accurate for small datasets

## Notes

A learning rate within  `[1e-2;1e-1]` seems like a good default.

The loss can sometimes decrease while the error is slightly increasing, this is due to the model fighting overfitting and improving its calibration.

Using SGD instead of Adam (`opt_func=optim.SGD`) is sometimes very beneficial with this kind of model but it can also lead to numerically unstable situations.

We provide two loss functions, other loss should take the mean *and* std into account.

The output has an std.

Singular matrix errors might happend due to numerical problems.

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
- add transfer learning (pretrained_model, freeze/unfreeze training data/kernel)
- add prior (zero, constant, linear, neural network)
- add active learning to select training points

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*