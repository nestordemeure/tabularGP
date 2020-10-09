# Tabular GP [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3723068.svg)](https://doi.org/10.5281/zenodo.3723068)

The aim of this repository is to make it easy to use gaussian process on tabular data, as a drop-in replacement for neural networks, with an implementation built on top of [pytorch](https://pytorch.org/) and the [fastai (V2) framework](https://docs.fast.ai/).

Gaussian process have three main properties that makes them of particular interest:
- they are very accurate, and tend to outperform deep neural network, on small datasets (5000 points or less)
- they gives an uncertainty estimate on their outputs
- they are naturally resistant to overfitting

If you are a gaussian process expert, you might be better served by [gpytorch](https://gpytorch.ai/) as our focus is on accesibility and out-of-the-box experiences rather than exhaustivity and flexibility.

## Usage

You can install our librarie with:

```
pip install git+https://github.com/nestordemeure/tabularGP.git
```

Our API was built to be compatible with [fastai's tabular models](https://docs.fast.ai/tabular.core) and should be familiar to fastai's users:

```python
# train a learner on a classification task using a subset of 50 points
learn = tabularGP_learner(data, nb_training_points=50, metrics=accuracy)
learn.fit_one_cycle(10, max_lr=1e-3)

# display the importance of each feature
learn.plot_feature_importance()
```

We recommand browsing the [example folder](https://github.com/nestordemeure/tabularGP/tree/master/examples) to become familiar with the features available (including various forms of transfer-learning and feature importance estimation) before using the librarie.

## Notes

The Gaussian process can produce a standard diviation to model the uncertainty on its output.
We store this information in a `.stdev` member of the output of out `forward` function in order to use it in the loss functions.
However, fastai erase this information when calling the `predict` function instead of calling `forward` directly.

Some inputs might lead to crash due to singular matrices appearing during the kernel computation.
The easiest solution to solve those problems is to restart the model with a lower learning rate (adding training points can also help).

We provide two loss functions out of the box (`gp_gaussian_marginal_log_likelihood` for regression and `gp_is_greater_log_likelihood` for classification).
Any user-defined loss function should take both mean and std into account to insure a proper fit.

One might observe that a validation metric increases while the loss steadily decreases.
This is due to the model improving its uncertainty estimate to the detriment of its prediction.

## Citation

If you use tabularGP in a scientific publication, you can cite the following reference:

```
@software{nestor_demeure_2020_3723068,
  author       = {Nestor Demeure},
  title        = {tabularGP},
  month        = mar,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3723068},
  url          = {https://doi.org/10.5281/zenodo.3723068}
}
```

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*
