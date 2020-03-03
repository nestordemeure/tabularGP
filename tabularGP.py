# Tabular GP
# Gaussian process based tabular model
# source: https://github.com/nestordemeure/tabularGP/blob/master/tabularGP.py

import numpy as np
import pandas
import torch
from torch import nn, Tensor
from fastai.tabular import DataBunch, ListSizes, ifnone, Learner
# my imports
from utils import psd_safe_cholesky, log_standard_normal_cdf
from kernel import ProductOfSumsKernel
from prior import ConstantPrior

__all__ = ['gp_gaussian_marginal_log_likelihood', 'gp_is_greater_log_likelihood',
           'TabularGPModel', 'TabularGPLearner', 'tabularGP_learner']

#--------------------------------------------------------------------------------------------------
# Training points selection

def _hamming_distances(row:Tensor, data:Tensor):
    "returns a vector with the hamming distance between a row and each row of a dataset"
    return (row.unsqueeze(dim=0) != data).sum(dim=1)

def _euclidian_distances(row:Tensor, data:Tensor):
    "returns a vector with the euclidian distance between a row and each row of a dataset"
    return torch.sum((row.unsqueeze(dim=0) - data)**2, dim=1)

def _maximalyDifferentPoints(data_cont:Tensor, data_cat:Tensor, nb_cluster:int):
    """
    returns the given number of indexes such that the associated rows are as far as possible
    according to the hamming distance between categories and, in case of equality, the euclidian distance between continuous columns
    uses a greedy algorithm to quickly get an approximate solution
    """
    # initialize with the first point of the dataset
    indexes = [0]
    row_cat = data_cat[0, ...]
    minimum_distances_cat = _hamming_distances(row_cat, data_cat)
    # we suppose that data_cont is normalized so raw euclidian distance is enough
    row_cont = data_cont[0, ...]
    minimum_distances_cont = _euclidian_distances(row_cont, data_cont)
    for _ in range(nb_cluster - 1):
        # finds the row that maximizes the minimum distances to the existing selections
        # choice is done on cat distance (which has granularity 1) and, in case of equality, cont distance (normalized to be in [0;0.5])
        minimum_distances = minimum_distances_cat + minimum_distances_cont / (2.0 * minimum_distances_cont.max())
        index = torch.argmax(minimum_distances, dim=0)
        indexes.append(index.item())
        # updates distances cont
        row_cont = data_cont[index, ...]
        distances_cont = _euclidian_distances(row_cont, data_cont)
        minimum_distances_cont = torch.min(minimum_distances_cont, distances_cont)
        # update distances cat
        row_cat = data_cat[index, ...]
        distances_cat = _hamming_distances(row_cat, data_cat)
        minimum_distances_cat = torch.min(minimum_distances_cat, distances_cat)
    return torch.LongTensor(indexes)

def _get_training_points(data:DataBunch, nb_points:int, use_random_training_points=False):
    "gets a (cat,cont,y) tuple with the given number of elements"
    # extracts all the dataset as a single tensor
    data_cat = []
    data_cont = []
    data_y = []
    for x,y in iter(data.train_dl):
        xcat = x[0]
        xcont = x[1]
        data_cat.append(xcat)
        data_cont.append(xcont)
        data_y.append(y)
    # concat the batches
    data_cat = torch.cat(data_cat)
    data_cont = torch.cat(data_cont)
    data_y = torch.cat(data_y)
    # transforms the output into one hot encoding if we are dealing with a classification problem
    is_classification = hasattr(data, 'classes')
    if is_classification: data_y = nn.functional.one_hot(data_y).float()
    # selects training points
    if nb_points >= data_cat.size(0): return (data_cat, data_cont, data_y)
    elif use_random_training_points: indices = torch.arange(0, nb_points)
    else: indices = _maximalyDifferentPoints(data_cont, data_cat, nb_points)
    # assemble the training data
    data_cat = data_cat[indices, ...]
    data_cont = data_cont[indices, ...]
    data_y = data_y[indices, ...]
    return (data_cat, data_cont, data_y)

#--------------------------------------------------------------------------------------------------
# Model

def gp_gaussian_marginal_log_likelihood(prediction, target:Tensor):
    "loss function for a regression gaussian process"
    mean = prediction
    stdev = prediction.stdev
    minus_log_likelihood = (mean - target)**2 / (2*stdev*stdev) + torch.log(stdev * np.sqrt(2.*np.pi))
    return minus_log_likelihood.mean()

def gp_is_greater_log_likelihood(prediction, target:Tensor):
    """
    compute the log probability that the target has a greater value than the other classes
     P(X>Y) = 0.5 * erfc(-mean / (sqrt(2)*std) )
     mean = mean_x - mean_y
     std² = std_x² + std_y² + 2*std_x*std_y
     under the hypothesis that corr(x,y) = -1 meaning that when one grows the other decreases
    for more information, see: https://math.stackexchange.com/questions/178334/the-probability-of-one-gaussian-larger-than-another
    """
    # gets the output distributions
    mean = prediction
    stdev = prediction.stdev
    # gets the target
    mean_target = mean[target]
    std_target = stdev[target]
    # computes the probability that the target is larger than another output
    mean_sub = mean_target - mean
    std_sub = torch.sqrt(std_target*std_target + stdev*stdev + 2.0*std_target*stdev)
    minus_log_proba = -log_standard_normal_cdf(mean_sub / std_sub)
    # removes the probability between the target and itself
    minus_log_proba[target] = 0.0
    minus_log_proba = torch.sum(minus_log_proba, dim=1)
    return minus_log_proba.mean()

class TabularGPModel(nn.Module):
    "Gaussian process based model for tabular data."
    def __init__(self, training_data:DataBunch, nb_training_points:int=50, use_random_training_points=False,
                 fit_training_inputs=True, fit_training_outputs=True, prior=ConstantPrior,
                 noise=1e-2, embedding_sizes:ListSizes=None, tabular_kernel=ProductOfSumsKernel, **kernel_kwargs):
        """
        'noise' is expressed as a fraction of the output std
        We recommend setting 'fit_training_outputs' to True for regression and False for classification
        """
        super().__init__()
        # registers training data
        train_input_cat, train_input_cont, train_outputs = _get_training_points(training_data, nb_training_points, use_random_training_points)
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
        embedding_sizes = training_data.get_emb_szs(ifnone(embedding_sizes, {}))
        self.kernel = tabular_kernel(train_input_cat, train_input_cont, embedding_sizes, **kernel_kwargs) if isinstance(tabular_kernel,type) else tabular_kernel
        self.prior = prior(train_input_cat, train_input_cont, train_outputs, embedding_sizes) if isinstance(prior,type) else prior

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        # covariance between combinaisons of samples
        cov_train_train = self.kernel.matrix((self.train_input_cat, self.train_input_cont), (self.train_input_cat, self.train_input_cont))
        cov_train_test = self.kernel.matrix((self.train_input_cat, self.train_input_cont), (x_cat, x_cont))
        diag_cov_test_test = self.kernel((x_cat, x_cont), (x_cat, x_cont))
        # cholesky decompositions (accelerate solving of linear systems)
        L_train_train = psd_safe_cholesky(cov_train_train)
        (L_test, _) = torch.triangular_solve(cov_train_test, L_train_train, upper=False)
        # outputs for the training data with prior correction
        train_outputs = self.train_outputs - self.prior(self.train_input_cat, self.train_input_cont)
        # predicted mean
        (output_to_weight, _) = torch.triangular_solve(train_outputs, L_train_train, upper=False)
        mean = torch.mm(L_test.t(), output_to_weight) + self.prior(x_cat, x_cont)
        # predicted std
        var_noise = self.std_noise * self.std_noise
        std_scale = self.std_scale.abs()
        covar = (diag_cov_test_test - torch.sum(L_test**2, dim=0)).abs().unsqueeze(dim=1) # abs against negative variance
        stdev = torch.sqrt(covar + var_noise) * std_scale + 1e-10 # epsilon to insure we are strictly above 0
        # adds std as an additional member to the mean
        mean.stdev = stdev
        return mean

    @property
    def feature_importance(self):
        return self.kernel.feature_importance

class TabularGPLearner(Learner):
    "Learner with some TabularGPModel specific methods"
    @property
    def feature_importance(self):
        "gets the feature importance for the model as a dataframe"
        # gets feature names
        cont_names = self.data.train_ds.x.cont_names
        cat_names = self.data.train_ds.x.cat_names
        feature_names = cat_names + cont_names
        # gets importance
        importances = self.model.feature_importance.detach().cpu()
        return pandas.DataFrame({'Variable': feature_names, 'Importance': importances})
    
    def plot_feature_importance(self, kind='barh', title="feature importance", figsize=(20,15), legend=False, **plot_kwargs):
        "produces a bar plot of the feature importance for the features, parameters are forwarded to the pandas plotting function"
        importances = self.feature_importance.sort_values('Importance')
        return importances.plot('Variable', 'Importance', kind=kind, title=title, figsize=figsize, legend=legend, **plot_kwargs)

def tabularGP_learner(data:DataBunch, nb_training_points:int=50, use_random_training_points=False,
                     fit_training_inputs=True, fit_training_outputs=None, prior=ConstantPrior,
                      noise=1e-2, embedding_sizes:ListSizes=None, tabular_kernel=ProductOfSumsKernel, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the associated data"
    # picks a loss function for the task
    # and decides wetehr training the outputs would give the best results
    is_classification = hasattr(data, 'classes')
    if is_classification:
        fit_training_outputs = False if fit_training_outputs is None else fit_training_outputs
        loss_func = gp_is_greater_log_likelihood
    else:
        fit_training_outputs = True if fit_training_outputs is None else fit_training_outputs
        loss_func = gp_gaussian_marginal_log_likelihood
    # defines the model
    model = TabularGPModel(training_data=data, nb_training_points=nb_training_points, use_random_training_points=use_random_training_points,
                           fit_training_inputs=fit_training_inputs, fit_training_outputs=fit_training_outputs, prior=prior,
                           noise=noise, embedding_sizes=embedding_sizes, tabular_kernel=tabular_kernel)
    return TabularGPLearner(data, model, loss_func=loss_func, **learn_kwargs)
