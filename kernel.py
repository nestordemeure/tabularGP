# Kernels
# Kernels for gaussian process based models
# a kernel can be seen as a function computing the similarity between two inputs
# source: https://github.com/nestordemeure/tabularGP/blob/master/kernel.py

import abc
# library imports
from fastai.tabular import ListSizes, TabularModel
import numpy as np
from torch import nn
import torch
# my imports
from utils import Scale
from universalCombinator import PositiveMultiply, PositiveProductOfSum

__all__ = ['CategorialKernel', 'ContinuousKernel', 'TabularKernel',
           'IndexKernelSingle', 'IndexKernel', 'HammingKernel',
           'GaussianKernel', 'ExponentialKernel', 'Matern1Kernel', 'Matern2Kernel', 'RBFKernel', 'MaternInfinityKernel', 'Matern0Kernel',
           'WeightedSumKernel', 'WeightedProductKernel', 'ProductOfSumsKernel', 'NeuralKernel']

#--------------------------------------------------------------------------------------------------
# abstract classes

class SingleColumnKernel(nn.Module):
    "Abstract class for kernels on single columns."
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x, y):
        "Computes the similarity between x and y, both being single columns."

    @property
    def feature_importance(self):
        return torch.ones(1).to(self.device)

class CategorialKernel(nn.Module):
    "Abstract class for kernels on categorial features."
    def __init__(self, embedding_sizes:ListSizes):
        super().__init__()
        self.nb_features = len(embedding_sizes)
        self.register_buffer('dummy', torch.empty(0))

    @property
    def device(self):
        "uses a dummy tensor to get the current device"
        return self.dummy.device

    @abc.abstractmethod
    def forward(self, x, y):
        "Computes the similarity between x and y, both being multi column categorial data."

    @property
    def feature_importance(self):
        return torch.ones(self.nb_features).to(self.device)


class ContinuousKernel(nn.Module):
    "Abstract class for kernels on continous features"
    def __init__(self, train_data):
        "Uses Silverman's rule of thumb as a default value for the bandwidth"
        super().__init__()
        default_bandwidth = 0.9 * train_data.std(dim=0) * (train_data.size(dim=0)**-0.2)
        self.bandwidth = nn.Parameter(default_bandwidth)
        self.nb_features = train_data.size(-1)

    @abc.abstractmethod
    def forward(self, x, y):
        "Computes the similarity between x and y, both being multi column continuous data."

    @property
    def feature_importance(self):
        return torch.ones(self.nb_features).to(self.bandwidth.device)

class TabularKernel(nn.Module):
    "abstract class for kernel applied to tabular data"
    def __init__(self, train_cat, train_cont, embedding_sizes:ListSizes):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x, y):
        "Computes the similarity between x and y, both being tuple of the form (cat,cont)."

    @property
    def feature_importance(self):
        raise Exception("The current tabular kernel does not implement feature importance.")

    def matrix(self, x, y):
        "Utilitary function that computes the matrix of all combinaison of kernel(x_i,y_j)"
        x_cat, x_cont = x
        y_cat, y_cont = y
        # cat
        nb_x_cat_elements = x_cat.size(0)
        nb_y_cat_elements = y_cat.size(0)
        cat_element_size = x_cat.size(1) if x_cat.dim() > 1 else 0
        if cat_element_size == 0: 
            # deals with the case where the is no cat feature
            x_cat = x_cat.unsqueeze(-1)
            y_cat = y_cat.unsqueeze(-1)
        x_cat = x_cat.unsqueeze(1).expand(nb_x_cat_elements, nb_y_cat_elements, cat_element_size)
        y_cat = y_cat.unsqueeze(0).expand(nb_x_cat_elements, nb_y_cat_elements, cat_element_size)
        # cont
        nb_x_cont_elements = x_cont.size(0)
        nb_y_cont_elements = y_cont.size(0)
        cont_element_size = x_cont.size(1) if x_cont.dim() > 1 else 0
        if cont_element_size == 0: 
            # deals with the case where the is no cont feature
            x_cont = x_cont.unsqueeze(-1)
            y_cont = y_cont.unsqueeze(-1)
        x_cont = x_cont.unsqueeze(1).expand(nb_x_cont_elements, nb_y_cont_elements, cont_element_size)
        y_cont = y_cont.unsqueeze(0).expand(nb_x_cont_elements, nb_y_cont_elements, cont_element_size)
        # covariance computation
        return self.forward((x_cat,x_cont), (y_cat,y_cont))

#--------------------------------------------------------------------------------------------------
# categorial kernels

class IndexKernelSingle(SingleColumnKernel):
    "IndexKernel but for a single column"
    def __init__(self, nb_category:int, rank:int, fraction_diagonal:float=0.9):
        "`fraction_diagonal` is used to set the initial weight repartition between the diagonal and the rest of the matrix"
        super().__init__()
        weight_sqrt_covar_factors = np.sqrt((1. - fraction_diagonal) / np.sqrt(rank)) # choosen so that the diagonal starts at 1
        self.sqrt_covar_factor = nn.Parameter(weight_sqrt_covar_factors * torch.ones((nb_category, rank)))
        self.std = nn.Parameter(fraction_diagonal * torch.ones(nb_category))

    def forward(self, x, y):
        "assumes that x and y have a single dimension"
        # uses the factors to build the covariance matrix
        covar_factor = self.sqrt_covar_factor * self.sqrt_covar_factor
        covariance = torch.mm(covar_factor, covar_factor.t())
        covariance.diagonal().add_(self.std*self.std)
        # evaluate the covariace matrix for our pairs of categories
        return covariance[x, y]

    @property
    def feature_importance(self):
        # uses the factors to build the covariance matrix
        covar_factor = self.sqrt_covar_factor * self.sqrt_covar_factor
        covariance = torch.mm(covar_factor, covar_factor.t())
        covariance.diagonal().add_(self.std*self.std)
        # the importance is the mean of the diagonal
        importance = covariance.diagonal().mean()
        return importance

class IndexKernel(CategorialKernel):
    """
    default kernel on categories
    inspired by [gpytorch's IndexKernel](https://gpytorch.readthedocs.io/en/latest/kernels.html#indexkernel)
    """
    def __init__(self, embedding_sizes:ListSizes):
        super().__init__(embedding_sizes)
        self.cat_covs = nn.ModuleList([IndexKernelSingle(nb_category,rank) for i,(nb_category,rank) in enumerate(embedding_sizes)])

    def forward(self, x, y):
        covariances = [cov(x[...,i],y[...,i]) for i,cov in enumerate(self.cat_covs)]
        if len(covariances) == 0: return torch.Tensor([]).to(x.device)
        return torch.stack(covariances, dim=-1)

    @property
    def feature_importance(self):
        importances = [cov.feature_importance for cov in self.cat_covs]
        if len(importances) == 0: return torch.Tensor([])
        return torch.stack(importances)

class HammingKernel(CategorialKernel):
    "trivial kernel on categories"
    def __init__(self, **args):
        super().__init__(**args)

    def forward(self, x, y):
        "1 where x=y, 0 otherwise"
        covariance = torch.zeros(x.shape).to(x.device)
        covariance[x == y] = 1.0
        return covariance

#--------------------------------------------------------------------------------------------------
# continuous kernels

class GaussianKernel(ContinuousKernel):
    "Default, gaussian, kernel"
    def forward(self, x, y):
        covariance = torch.exp( -(x - y)**2 / (2 * self.bandwidth * self.bandwidth) )
        return covariance

class ExponentialKernel(ContinuousKernel):
    "Exponential kernel (matern 1/2), zero differentiable"
    def forward(self, x, y):
        covariance = torch.exp(-torch.abs(x - y) / self.bandwidth)
        return covariance

class Matern1Kernel(ContinuousKernel):
    "Matern 3/2 kernel, once differentiable"
    def forward(self, x, y):
        d = torch.abs(x - y)
        term = np.sqrt(3) * d / self.bandwidth
        covariance = (1 + term) * torch.exp(-term)
        return covariance

class Matern2Kernel(ContinuousKernel):
    "Matern 5/2 kernel, twice differentiable"
    def forward(self, x, y):
        d = torch.abs(x - y)
        term = np.sqrt(5) * d / self.bandwidth
        covariance = (1 + term + term*term/3) * torch.exp(-term)
        return covariance

# aliases for various kernel
RBFKernel = GaussianKernel
MaternInfinityKernel = GaussianKernel
Matern0Kernel = ExponentialKernel

#--------------------------------------------------------------------------------------------------
# tabular kernels

class WeightedSumKernel(TabularKernel):
    "Minimal kernel for tabular data, sums the covariances for all the columns"
    def __init__(self, train_cat, train_cont, embedding_sizes:ListSizes, cont_kernel=GaussianKernel, cat_kernel=IndexKernel):
        super().__init__(train_cat, train_cont, embedding_sizes)
        self.cont_kernel = cont_kernel(train_cont)
        self.cat_kernel = cat_kernel(embedding_sizes)
        nb_cat_features = train_cat.size(1) if train_cat.dim() > 1 else 0
        nb_cont_features = train_cont.size(1) if train_cont.dim() > 1 else 0
        self.scale = Scale(nb_cat_features+nb_cont_features)

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        # computes individual covariances
        x_cat, x_cont = x
        y_cat, y_cont = y
        covariances = torch.cat((self.cont_kernel(x_cont, y_cont), self.cat_kernel(x_cat, y_cat)), dim=-1)
        # weighted sum of the covariances
        covariance = torch.sum(self.scale(covariances), dim=-1)
        return covariance

    @property
    def feature_importance(self):
        imp_cat = self.cat_kernel.feature_importance
        imp_cont = self.cont_kernel.feature_importance
        importances = self.scale(torch.cat([imp_cat, imp_cont], dim=-1))
        return importances

class WeightedProductKernel(TabularKernel):
    "Learns a weighted geometric average of the covariances for all the columns"
    def __init__(self, train_cat, train_cont, embedding_sizes:ListSizes, cont_kernel=GaussianKernel, cat_kernel=IndexKernel):
        super().__init__(train_cat, train_cont, embedding_sizes)
        self.cont_kernel = cont_kernel(train_cont)
        self.cat_kernel = cat_kernel(embedding_sizes)
        nb_cat_features = train_cat.size(1) if train_cat.dim() > 1 else 0
        nb_cont_features = train_cont.size(1) if train_cont.dim() > 1 else 0
        self.combinator = PositiveMultiply(in_features=nb_cat_features+nb_cont_features, out_features=1, bias=False)

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        x_cat, x_cont = x
        y_cat, y_cont = y
        covariances = torch.cat((self.cont_kernel(x_cont, y_cont), self.cat_kernel(x_cat, y_cat)), dim=-1)
        covariance = self.combinator(covariances).squeeze(dim=-1)
        return covariance

    @property
    def feature_importance(self):
        # collect individual feature importances
        imp_cat = self.cat_kernel.feature_importance
        imp_cont = self.cont_kernel.feature_importance
        importances = torch.cat([imp_cat, imp_cont], dim=-1)
        # value with feature minus importances without feature
        baseline_importance = self.combinator(importances)
        nb_features = importances.size(0)
        importances = importances.repeat(nb_features, 1)
        importances.fill_diagonal_(0.0)
        importances = baseline_importance - self.combinator(importances)
        return importances.squeeze()

class ProductOfSumsKernel(TabularKernel):
    "Learns an arbitrary weighted geometric average of the sum of the covariances for all the columns."
    def __init__(self, train_cat, train_cont, embedding_sizes:ListSizes, cont_kernel=GaussianKernel, cat_kernel=IndexKernel):
        super().__init__(train_cat, train_cont, embedding_sizes)
        self.cont_kernel = cont_kernel(train_cont)
        self.cat_kernel = cat_kernel(embedding_sizes)
        nb_cat_features = train_cat.size(1) if train_cat.dim() > 1 else 0
        nb_cont_features = train_cont.size(1) if train_cont.dim() > 1 else 0
        self.combinator = PositiveProductOfSum(in_features=nb_cat_features+nb_cont_features, out_features=1)

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        x_cat, x_cont = x
        y_cat, y_cont = y
        covariances = torch.cat((self.cont_kernel(x_cont, y_cont), self.cat_kernel(x_cat, y_cat)), dim=-1)
        covariance = self.combinator(covariances).squeeze(dim=-1)
        return covariance

    @property
    def feature_importance(self):
        # collect individual feature importances
        imp_cat = self.cat_kernel.feature_importance
        imp_cont = self.cont_kernel.feature_importance
        importances = torch.cat([imp_cat, imp_cont], dim=-1)
        # value with feature minus importances without feature
        baseline_importance = self.combinator(importances)
        nb_features = importances.size(0)
        importances = importances.repeat(nb_features, 1)
        importances.fill_diagonal_(0.0)
        importances = baseline_importance - self.combinator(importances)
        return importances.squeeze()

class NeuralKernel(TabularKernel):
    "Uses a neural network to learn an embedding for the inputs. The covariance between two inputs is their cosinus similarity."
    def __init__(self, train_cat, train_cont, embedding_sizes:ListSizes, neural_embedding_size:int=20, layers=[200,100], **neuralnetwork_kwargs):
        super().__init__(train_cat, train_cont, embedding_sizes)
        self.encoder = TabularModel(emb_szs=embedding_sizes, n_cont=train_cont.size(-1), out_sz=neural_embedding_size, layers=layers, y_range=None, **neuralnetwork_kwargs)
        self.scale = Scale(neural_embedding_size)

    def kernel(self, x, y):
        "RBF type of kernel"
        covariance = self.scale(torch.exp(-(x - y)**2)) # no bandwith as we found it detrimental
        return torch.sum(covariance, dim=-1)

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        x = self.encoder(*x)
        y = self.encoder(*y)
        return self.kernel(x,y)

    def matrix(self, x, y):
        "Computes the matrix of all combinaison of kernel(x_i,y_j)"
        # encodes the inputs
        x = self.encoder(*x)
        y = self.encoder(*y)
        # builds matrix
        nb_x_elements = x.size(0)
        nb_y_elements = y.size(0)
        element_size = x.size(1)
        x = x.unsqueeze(1).expand(nb_x_elements, nb_y_elements, element_size)
        y = y.unsqueeze(0).expand(nb_x_elements, nb_y_elements, element_size)
        return self.kernel(x,y)
