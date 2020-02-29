# Kernels
# Kernels for gaussian process based models
# a kernel can be seen as a fucntion computing the similarity between two inputs
# source: https://github.com/nestordemeure/tabularGP/blob/master/kernel.py

from fastai.tabular import ListSizes
import numpy as np
from torch import nn
import torch
# my imports
from universalCombinator import PositiveProductOfSum

__all__ = ['kernelMatrix', 'IndexKernelSingle', 'RBFKernel', 'HammingKernel', 'IndexKernel', 'WeightedSumKernel']

#--------------------------------------------------------------------------------------------------
# functions

def kernelMatrix(kernel, x, y):
    "Utilitary function that computes the matrix of all combinaison of kernel(x_i,y_j)"
    x_cat, x_cont = x
    y_cat, y_cont = y
    # cat
    nb_x_cat_elements = x_cat.size(0)
    nb_y_cat_elements = y_cat.size(0)
    cat_element_size = x_cat.size(1)
    x_cat = x_cat.unsqueeze(1).expand(nb_x_cat_elements, nb_y_cat_elements, cat_element_size)
    y_cat = y_cat.unsqueeze(0).expand(nb_x_cat_elements, nb_y_cat_elements, cat_element_size)
    # cont
    nb_x_cont_elements = x_cont.size(0)
    nb_y_cont_elements = y_cont.size(0)
    cont_element_size = x_cont.size(1)
    x_cont = x_cont.unsqueeze(1).expand(nb_x_cont_elements, nb_y_cont_elements, cont_element_size)
    y_cont = y_cont.unsqueeze(0).expand(nb_x_cont_elements, nb_y_cont_elements, cont_element_size)
    # covariance computation
    return kernel((x_cat,x_cont), (y_cat,y_cont))

def _default_bandwidth(x):
    "Silverman's rule of thumb as a default value for the bandwidth"
    return 0.9 * x.std(dim=0) * (x.size(dim=0)**-0.2)

#--------------------------------------------------------------------------------------------------
# single column kernels

class IndexKernelSingle(nn.Module):
    "IndexKernel but for a single column"
    def __init__(self, train_data, nb_category:int, rank:int, fraction_diagonal:float=0.9):
        "`fraction_diagonal` is used to set the initial weight repartition between the diagonal and the rest of the matrix"
        super().__init__()
        self.std = nn.Parameter(fraction_diagonal * torch.ones(nb_category))
        self.covar_factor = nn.Parameter(((1. - fraction_diagonal) / np.sqrt(rank)) * torch.ones((nb_category, rank)))

    def forward(self, x, y):
        "assumes that x and y have a single dimension"
        # uses the factors to build the covariance matrix
        covariance = torch.mm(self.covar_factor, self.covar_factor.t())
        covariance.diagonal().add_(self.std*self.std)
        # evaluate the covariace matrix for our pairs of categories
        return covariance[x, y]

#--------------------------------------------------------------------------------------------------
# multi columns kernels

class RBFKernel(nn.Module):
    "default, gaussian, kernel on reals"
    def __init__(self, train_data, should_reduce=True):
        super().__init__()
        self.should_reduce = should_reduce
        self.nb_training_points = train_data.size(1)
        self.bandwidth = nn.Parameter(_default_bandwidth(train_data))
        self.sqrt_scale = nn.Parameter(torch.ones(train_data.size(1)))

    def forward(self, x, y):
        scale = (self.sqrt_scale * self.sqrt_scale).unsqueeze(dim=0)
        covariance = scale * torch.exp( -(x - y)**2 / (2 * self.bandwidth * self.bandwidth).unsqueeze(dim=0) )
        if self.should_reduce: return covariance.sum(dim=-1)
        else: return covariance

class HammingKernel(nn.Module):
    "trivial kernel on categories"
    def __init__(self, train_data, should_reduce=True):
        super().__init__()
        self.should_reduce = should_reduce
        self.nb_training_points = train_data.size(1)
        self.sqrt_scale = nn.Parameter(torch.ones(train_data.size(1)))

    def forward(self, x, y):
        "1 where x=y, 0 otherwise"
        covariance = torch.zeros(x.shape).to(x.device)
        covariance[x == y] = 1.0
        scale = (self.sqrt_scale * self.sqrt_scale).unsqueeze(dim=0)
        covariance = scale * covariance
        if self.should_reduce: return covariance.sum(dim=-1)
        else: return covariance

class IndexKernel(nn.Module):
    """
    default kernel on categories
    inspired by [gpytorch's IndexKernel](https://gpytorch.readthedocs.io/en/latest/kernels.html#indexkernel)
    """
    def __init__(self, train_data, embedding_sizes:ListSizes, should_reduce=True):
        super().__init__()
        self.should_reduce = should_reduce
        self.cat_covs = nn.ModuleList([IndexKernelSingle(train_data[:,i],nb_category,rank) for i,(nb_category,rank) in enumerate(embedding_sizes)])

    def forward(self, x, y):
        covariances = [cov(x[...,i],y[...,i]) for i,cov in enumerate(self.cat_covs)]
        if self.should_reduce: return sum(covariances)
        else: return torch.stack(covariances, dim=-1)

#--------------------------------------------------------------------------------------------------
# tabular kernels

class WeightedSumKernel(nn.Module):
    "Minimal kernel for tabular data, sums the kernel for all the columns"
    def __init__(self, train_cont, train_cat, embedding_sizes:ListSizes):
        super().__init__()
        self.cont_kernel = RBFKernel(train_cont)
        self.cat_kernel = IndexKernel(train_cat, embedding_sizes)

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        x_cat, x_cont = x
        y_cat, y_cont = y
        covariance = self.cont_kernel(x_cont, y_cont) + self.cat_kernel(x_cat, y_cat)
        return covariance

# TODO add possibility to deactivate scaling on kernels
# TODO add neural network encoder kernel
# TODO add Weighted Product kernel
# TODO add universa combinator kernel