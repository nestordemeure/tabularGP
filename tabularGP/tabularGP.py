# Tabular GP
# Gaussian process based tabular model
# source: https://github.com/nestordemeure/tabularGP/blob/master/tabularGP.py

import pandas
import torch
from torch import nn, Tensor
from fastai.tabular.all import ifnone, Learner, get_emb_sz
# my imports
from tabularGP.loss_functions import gp_gaussian_marginal_log_likelihood, gp_is_greater_log_likelihood, gp_metric_wrapper
from tabularGP.utils import psd_safe_cholesky, freeze, unfreeze
from tabularGP.kernel import ProductOfSumsKernel, TabularKernel
from tabularGP.trainset_selection import select_trainset
from tabularGP.prior import ConstantPrior

__all__ = ['TabularGPModel', 'TabularGPLearner', 'tabularGP_learner']

#--------------------------------------------------------------------------------------------------
# Model

class TabularGPModel(nn.Module):
    "Gaussian process based model for tabular data."
    def __init__(self, training_data, nb_training_points:int=4000, use_random_training_points=False,
                 fit_training_inputs=False, fit_training_outputs=False, prior=ConstantPrior,
                 noise=1e-2, embedding_sizes=None, kernel=ProductOfSumsKernel, **kernel_kwargs):
        """
        'noise' is expressed as a fraction of the output std
        We recommend setting 'fit_training_outputs' to True for regression and False for classification
        """
        super().__init__()
        # registers training data
        train_input_cat, train_input_cont, train_outputs = select_trainset(training_data, nb_training_points, use_random_training_points)
        if train_outputs.dim() == 1: train_outputs = train_outputs.unsqueeze(dim=-1) # deals with 1D outputs
        self.register_buffer('train_input_cat', train_input_cat)
        if fit_training_inputs: self.train_input_cont = nn.Parameter(train_input_cont)
        else: self.register_buffer('train_input_cont', train_input_cont)
        if fit_training_outputs: self.train_outputs = nn.Parameter(train_outputs)
        else: self.register_buffer('train_outputs', train_outputs)
        # kernel and associated parameters
        output_std = train_outputs.std(dim=0)
        self.std_scale = nn.Parameter(output_std)
        self.std_noise = nn.Parameter(output_std * noise)
        # TODO find function to get embeddings sizes for data
        embedding_sizes = get_emb_sz(training_data.train_ds, {} if embedding_sizes is None else embedding_sizes)
        self.kernel = kernel(train_input_cat, train_input_cont, embedding_sizes, **kernel_kwargs) if isinstance(kernel,type) else kernel
        self.prior = prior(train_input_cat, train_input_cont, train_outputs, embedding_sizes) if isinstance(prior,type) else prior
        # precomputed cholesky decomposition
        self._L_train_train = None
        self._output_weights = None

    def memoized_cholesky_decomposition(self):
        "memoize the cholesky decomposition to avoid recomputing it when we are not training"
        if (self._L_train_train is None) or (self.training):
            # covariance between training samples
            cov_train_train = self.kernel.matrix((self.train_input_cat, self.train_input_cont), (self.train_input_cat, self.train_input_cont))
            self._L_train_train = psd_safe_cholesky(cov_train_train).detach() # we drop the gradient for the cholesky decomposition
            # outputs for the training data with prior correction
            train_outputs = self.train_outputs - self.prior(self.train_input_cat, self.train_input_cont)
            # weights for the predicted mean
            self._output_weights, _ = torch.triangular_solve(train_outputs, self._L_train_train, upper=False)
        return self._L_train_train, self._output_weights

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        # covariance between combinaisons of samples
        cov_train_test = self.kernel.matrix((self.train_input_cat, self.train_input_cont), (x_cat, x_cont))
        diag_cov_test_test = self.kernel((x_cat, x_cont), (x_cat, x_cont))
        # cholesky decompositions (accelerate solving of linear systems)
        L_train_train, output_weights = self.memoized_cholesky_decomposition()
        (L_test, _) = torch.triangular_solve(cov_train_test, L_train_train, upper=False)
        # predicted mean
        mean = torch.mm(L_test.t(), output_weights) + self.prior(x_cat, x_cont)
        # predicted std
        var_noise = self.std_noise * self.std_noise
        std_scale = self.std_scale.abs()
        covar = (diag_cov_test_test - torch.sum(L_test**2, dim=0)).abs().unsqueeze(dim=1) # abs against negative variance
        stdev = torch.sqrt(covar + var_noise) * std_scale + 1e-10 # epsilon to insure we are strictly above 0
        # builds a tensor with the mean and std information stored in the last dimenssion
        prediction = torch.stack([mean, stdev], dim=-1)
        return prediction

    @property
    def feature_importance(self):
        return self.kernel.feature_importance

#--------------------------------------------------------------------------------------------------
# Learner

class TabularGPLearner(Learner):
    "Learner with some TabularGPModel specific methods"
    def __init__(self, data, model, metrics=None, **kwargs):
        # wrapper to make output type compatible with classical metrics
        wrapped_metrics = gp_metric_wrapper(metrics)
        super().__init__(data, model, metrics=wrapped_metrics, **kwargs)

    @property
    def feature_importance(self):
        "gets the feature importance for the model as a dataframe"
        # gets feature names
        cont_names = self.data.cont_names
        cat_names = self.data.cat_names
        feature_names = cat_names + cont_names
        # gets importance
        importances = self.model.feature_importance.detach().cpu()
        return pandas.DataFrame({'Variable': feature_names, 'Importance': importances})

    def plot_feature_importance(self, kind='barh', title="Feature importance", figsize=(20,15), legend=False, **plot_kwargs):
        "produces a bar plot of the feature importance for the features, parameters are forwarded to the pandas plotting function"
        importances = self.feature_importance.sort_values('Importance')
        return importances.plot('Variable', 'Importance', kind=kind, title=title, figsize=figsize, legend=legend, **plot_kwargs)

    def freeze(self, kernel=None, data=None, prior=None, covar_scaling=None)->None:
        "freeze all the value (if they are all none) or only the one set to true"
        allnone = (kernel is None) and (prior is None) and (data is None) and (covar_scaling is None)
        if kernel or allnone:
            freeze(self.model.kernel)
        if prior or allnone:
            freeze(self.model.prior)
        if data or allnone:
            freeze(self.model.train_input_cont)
            freeze(self.model.train_outputs)
        if covar_scaling or allnone:
            freeze(self.model.std_scale)
            freeze(self.model.std_noise)
        self.create_opt()

    def unfreeze(self, kernel=None, data=None, prior=None, covar_scaling=None)->None:
        "freeze all the value (if they are all none) or only the one set to true"
        allnone = (kernel is None) and (prior is None) and (data is None) and (covar_scaling is None)
        if kernel or allnone:
            unfreeze(self.model.kernel)
        if prior or allnone:
            unfreeze(self.model.prior)
        if data or allnone:
            unfreeze(self.model.train_input_cont)
            unfreeze(self.model.train_outputs)
        if covar_scaling or allnone:
            unfreeze(self.model.std_scale)
            unfreeze(self.model.std_noise)
        self.create_opt()

#--------------------------------------------------------------------------------------------------
# Constructor

def tabularGP_learner(data, nb_training_points:int=4000, use_random_training_points=False,
                     fit_training_inputs=False, fit_training_outputs=False, prior=ConstantPrior,
                     noise=1e-2, embedding_sizes=None, kernel=ProductOfSumsKernel, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the associated data"
    # loss function
    # also decides whethere training the outputs would give the best results
    is_classification = (data.c > 1) and (len(data.y_names) == 1)
    if is_classification: loss_func = gp_is_greater_log_likelihood
    else: loss_func = gp_gaussian_marginal_log_likelihood
    # kernel
    if not isinstance(kernel, type):
        if isinstance(kernel, TabularGPLearner): kernel = kernel.model.kernel
        elif isinstance(kernel, TabularGPModel): kernel = kernel.kernel
        elif not isinstance(kernel, TabularKernel): raise Exception("kernel type unrecognized. Please use a TabularGPLearner, TabularGPModel or TabularKernel")
        freeze(kernel) # freezes kernel when doing transfer learning
    # prior
    if isinstance(prior, Learner): prior = prior.model
    if not isinstance(prior, type):
        freeze(prior) # freezes prior when doing transfer learning
    # defines the model
    model = TabularGPModel(training_data=data, nb_training_points=nb_training_points, use_random_training_points=use_random_training_points,
                           fit_training_inputs=fit_training_inputs, fit_training_outputs=fit_training_outputs,
                           prior=prior, noise=noise, embedding_sizes=embedding_sizes, kernel=kernel)
    return TabularGPLearner(data, model, loss_func=loss_func, **learn_kwargs)
