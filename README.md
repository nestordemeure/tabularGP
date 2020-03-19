# Tabular GP

The aim of this repository is to make it easy to use gaussian process on tabular data with an implementation built on top of [pytorch](https://pytorch.org/) and the [fastai V1 framework](https://docs.fast.ai/).

Gaussian process have three main properties that makes them of particular interest:
- they are very accurate, and tend to outperform deep neural network, on small datasets (5000 points or less)
- they gives an uncertainty estimate on their outputs
- they are naturally resistant to overfitting

If you are a gaussian process expert, you might be better served by [gpytorch](https://gpytorch.ai/) as our focus is on accesibility and out-of-the-box experiences rather than exhaustivity and flexibility.

## Usage

Our API was built to be compatible with [fastai V1's tabular models](https://docs.fast.ai/tabular.html) and should be familiar to fastai's users:

```python
# train a learner on a classification task using a subset of 50 points
learn = tabularGP_learner(data, nb_training_points=50, metrics=accuracy)
learn.fit_one_cycle(10, max_lr=1e-3)

# display the feature importance
glearn.plot_feature_importance()
```

For a tour of the features available (including various forms of transfer-learning and feature importance estimation), see the [example folder](TODO).

## Notes

The gaussian process output a tensor, its prediction (the *mean* of the gaussian process), with an additional `stdev` member to get the uncertainty on the prediction.

Some inputs might lead to crash due to singular matrices appearing during the kernel computation.
The easiest solution to those problems is to restart the model with a lower learning rate (adding training points can also help).

We provide two loss functions out of the box (`gp_gaussian_marginal_log_likelihood` for regression and `gp_is_greater_log_likelihood` for classification). Any user-defined loss function should take both mean and std into account to insure a proper fit.

One might observe that a validation metric increases while the loss steadily decreases.
This is due to the model improving its uncertainty estimate to the detriment of its prediction.

## TODO

- make example folder covering basic usage, feature importance, transfer learning, kernel selection
- submit to pip
- add a DOI for ease of quote in papers

- add a date/time specific kernel (periodic kernel)
- add possibility to pass a list of kernels to the tabular kernels (to have one specific kernel per column)

- explore other loss functions for classification (softmax)

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*