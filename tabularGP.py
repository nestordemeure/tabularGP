# Tabular GP
# Gaussian process based tabular model
# source: https://github.com/nestordemeure/tabularGP/blob/master/tabularGP.py

import numpy as np
import torch
from torch import nn, Tensor
from fastai.tabular import DataBunch, ListSizes, ifnone, Learner
# my imports
from utils import psd_safe_cholesky
from kernel import kernelMatrix, WeightedSumKernel

__all__ = ['gp_gaussian_marginal_log_likelihood', 'TabularGPModel', 'tabularGP_learner']

def _get_training_points(data:DataBunch, nb_points:int):
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
    # selects inducing points
    data_cat = data_cat[:nb_points, :]
    data_cont = data_cont[:nb_points, :]
    data_y = data_y[:nb_points, :]
    return (data_cat, data_cont, data_y)

def gp_gaussian_marginal_log_likelihood(prediction, target:Tensor):
    "loss function for a regression gaussian process"
    mean = prediction
    stdev = prediction.stdev
    minus_log_likelihood = (mean - target)**2 / (2*stdev*stdev) + torch.log(stdev * np.sqrt(2.*np.pi))
    return minus_log_likelihood.mean()

class TabularGPModel(nn.Module):
    "Gaussian process based model for tabular data."
    def __init__(self, training_data:DataBunch, nb_training_points:int=50, fit_training_point=True, noise=1e-2, embedding_sizes:ListSizes=None):
        "noise is in faction of the output std"
        super().__init__()
        # registers training data
        train_input_cat, train_input_cont, train_outputs = _get_training_points(training_data, nb_training_points)
        self.register_buffer('train_input_cat', train_input_cat)
        if fit_training_point:
            self.train_input_cont = nn.Parameter(train_input_cont)
            self.train_outputs = nn.Parameter(train_outputs)
        else:
            self.register_buffer('train_input_cont', train_input_cont)
            self.register_buffer('train_outputs', train_outputs)
        # kernel and associated parameters
        output_std = train_outputs.std(dim=0)
        self.std_scale = nn.Parameter(output_std)
        self.std_noise = nn.Parameter(output_std * noise)
        embedding_sizes = training_data.get_emb_szs(ifnone(embedding_sizes, {}))
        self.kernel = WeightedSumKernel(train_input_cont, train_input_cat, embedding_sizes)
        self.prior = nn.Parameter(train_outputs.mean(dim=0)) # constant prior

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        # covariance between combinaisons of samples
        cov_train_train = kernelMatrix(self.kernel, (self.train_input_cat, self.train_input_cont), (self.train_input_cat, self.train_input_cont))
        cov_train_test = kernelMatrix(self.kernel, (self.train_input_cat, self.train_input_cont), (x_cat, x_cont))
        diag_cov_test_test = self.kernel((x_cat, x_cont), (x_cat, x_cont))
        # cholesky decompositions (accelerate solving of linear systems)
        L_train_train = psd_safe_cholesky(cov_train_train)
        (L_test, _) = torch.triangular_solve(cov_train_test, L_train_train, upper=False)
        # predicted mean
        prior = self.prior.unsqueeze(dim=0)
        (output_to_weight, _) = torch.triangular_solve(self.train_outputs - prior, L_train_train, upper=False) 
        mean = torch.mm(L_test.t(), output_to_weight) + prior
        # predicted std
        var_noise = (self.std_noise * self.std_noise).clamp_min(1e-8).unsqueeze(dim=0) # clamp to insure we are above 0
        std_scale = torch.abs(self.std_scale).unsqueeze(dim=0)
        covar = (diag_cov_test_test - torch.sum(L_test**2, dim=0)).clamp_min(0.0).unsqueeze(dim=1) # clamp against negative variance
        stdev = torch.sqrt(covar + var_noise) * std_scale
        # adds std as an additional member to the mean
        mean.stdev = stdev
        return mean

def tabularGP_learner(data:DataBunch, nb_training_points:int=50, fit_training_point=True, noise=1e-2, embedding_sizes:ListSizes=None, metrics=None, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the associated data"
    # picks a loss function for the task
    is_classification = hasattr(data, 'classes')
    if is_classification: raise Exception("tabularGP does not implement classification yet!")
    else: loss_func = gp_gaussian_marginal_log_likelihood
    # defines the model
    model = TabularGPModel(data, nb_training_points, fit_training_point, noise, embedding_sizes)
    return Learner(data, model, metrics=metrics, loss_func=loss_func, **learn_kwargs)

# TODO add methods to do classification
# TODO add feature importance for kernels that support it out of the box
# TODO add transfer learning (by reusing the kernel)
# TODO use kmean clustering to find representative inducing points
