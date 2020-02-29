# Kernels
# Kernels for gaussian process based models
# a kernel can be seen as a fucntion computing the similarity between two inputs
# source: https://github.com/nestordemeure/tabularGP/blob/master/kernel.py

from fastai.tabular import ListSizes, TabularModel
import numpy as np
from torch import nn
import torch
# my imports
from universalCombinator import PositiveMultiply, PositiveProductOfSum

__all__ = ['IndexKernelSingle', 'IndexKernel', 'HammingKernel', 'RBFKernel',
           'WeightedSumKernel', 'WeightedProductKernel', 'ProductOfSumsKernel', 'NeuralKernel']

#--------------------------------------------------------------------------------------------------
# categorial kernels

class IndexKernelSingle(nn.Module):
    "IndexKernel but for a single column"
    def __init__(self, train_data, nb_category:int, rank:int, fraction_diagonal:float=0.9):
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

class IndexKernel(nn.Module):
    """
    default kernel on categories
    inspired by [gpytorch's IndexKernel](https://gpytorch.readthedocs.io/en/latest/kernels.html#indexkernel)
    """
    def __init__(self, train_data, embedding_sizes:ListSizes):
        super().__init__()
        self.cat_covs = nn.ModuleList([IndexKernelSingle(train_data[:,i],nb_category,rank) for i,(nb_category,rank) in enumerate(embedding_sizes)])

    def forward(self, x, y):
        covariances = [cov(x[...,i],y[...,i]) for i,cov in enumerate(self.cat_covs)]
        return torch.stack(covariances, dim=-1)

class HammingKernel(nn.Module):
    "trivial kernel on categories"
    def __init__(self, train_data):
        super().__init__()
        self.nb_training_points = train_data.size(1)

    def forward(self, x, y):
        "1 where x=y, 0 otherwise"
        covariance = torch.zeros(x.shape).to(x.device)
        covariance[x == y] = 1.0
        return covariance

#--------------------------------------------------------------------------------------------------
# continuous kernels

def _default_bandwidth(x):
    "Silverman's rule of thumb as a default value for the bandwidth"
    return 0.9 * x.std(dim=0) * (x.size(dim=0)**-0.2)

class RBFKernel(nn.Module):
    "default, gaussian, kernel on reals"
    def __init__(self, train_data):
        super().__init__()
        self.nb_training_points = train_data.size(1)
        self.bandwidth = nn.Parameter(_default_bandwidth(train_data))

    def forward(self, x, y):
        covariance = torch.exp( -(x - y)**2 / (2 * self.bandwidth * self.bandwidth).unsqueeze(dim=0) )
        return covariance

class CosineSimilarityKernel(nn.Module):
    "performs a cosine similarity between all the columns at once"
    def forward(self, x, y):
        similarity = nn.functional.cosine_similarity(x, y, dim=-1) # in [-1,1]
        covariance = 0.5 + 0.5*similarity
        return covariance

#--------------------------------------------------------------------------------------------------
# tabular kernels

class TabularKernel(nn.Module):
    "abstract class for kernel applied to tabular data"
    def __init__(self, train_cont, train_cat, embedding_sizes:ListSizes):
        super().__init__()

    def matrix(self, x, y):
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
        return self.forward((x_cat,x_cont), (y_cat,y_cont))

#------------------------------------------------

class WeightedSumKernel(TabularKernel):
    "Minimal kernel for tabular data, sums the covariances for all the columns"
    def __init__(self, train_cont, train_cat, embedding_sizes:ListSizes, cont_kernel=RBFKernel, cat_kernel=IndexKernel):
        super().__init__(train_cont, train_cat, embedding_sizes)
        self.cont_kernel = cont_kernel(train_cont)
        self.cat_kernel = cat_kernel(train_cat, embedding_sizes)
        nb_features = train_cont.size(1) + train_cat.size(1)
        self.sqrt_scale = nn.Parameter(torch.ones(nb_features))

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        # computes individual covariances
        x_cat, x_cont = x
        y_cat, y_cont = y
        covariances = torch.cat((self.cont_kernel(x_cont, y_cont), self.cat_kernel(x_cat, y_cat)), dim=-1)
        # weihgted sum of the covariances
        scale = (self.sqrt_scale * self.sqrt_scale).unsqueeze(dim=0)
        covariance = torch.sum(scale * covariances, dim=-1)
        return covariance

class WeightedProductKernel(TabularKernel):
    "Learns a weighted geometric average of the covariances for all the columns"
    def __init__(self, train_cont, train_cat, embedding_sizes:ListSizes, cont_kernel=RBFKernel, cat_kernel=IndexKernel):
        super().__init__(train_cont, train_cat, embedding_sizes)
        self.cont_kernel = cont_kernel(train_cont)
        self.cat_kernel = cat_kernel(train_cat, embedding_sizes)
        nb_features = train_cont.size(1) + train_cat.size(1)
        self.combinator = PositiveMultiply(in_features=nb_features, out_features=1, bias=False)

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        x_cat, x_cont = x
        y_cat, y_cont = y
        covariances = torch.cat((self.cont_kernel(x_cont, y_cont), self.cat_kernel(x_cat, y_cat)), dim=-1)
        covariance = self.combinator(covariances).squeeze(dim=-1)
        return covariance

class ProductOfSumsKernel(TabularKernel):
    "Learns an arbitrary weighted geometric average of the sum of the covariances for all the columns"
    def __init__(self, train_cont, train_cat, embedding_sizes:ListSizes, cont_kernel=RBFKernel, cat_kernel=IndexKernel):
        super().__init__(train_cont, train_cat, embedding_sizes)
        self.cont_kernel = cont_kernel(train_cont)
        self.cat_kernel = cat_kernel(train_cat, embedding_sizes)
        nb_features = train_cont.size(1) + train_cat.size(1)
        self.combinator = PositiveProductOfSum(in_features=nb_features, out_features=1)

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        x_cat, x_cont = x
        y_cat, y_cont = y
        covariances = torch.cat((self.cont_kernel(x_cont, y_cont), self.cat_kernel(x_cat, y_cat)), dim=-1)
        covariance = self.combinator(covariances).squeeze(dim=-1)
        return covariance

class NeuralKernel(TabularKernel):
    "Uses a neural network to learn an embedding for the inputs. The covariance between two inputs is their cosinus similarity."
    def __init__(self, train_cont, train_cat, embedding_sizes:ListSizes, neural_embedding_size:int=20, layers=[200,100], **neuralnetwork_kwargs):
        super().__init__(train_cont, train_cat, embedding_sizes)
        self.encoder = TabularModel(emb_szs=embedding_sizes, n_cont=train_cont.size(-1), out_sz=neural_embedding_size, layers=layers, y_range=None, **neuralnetwork_kwargs)
        self.kernel = CosineSimilarityKernel()

    def forward(self, x, y):
        "returns a tensor with one similarity per pair (x_i,y_i) of batch element"
        x = self.encoder(*x)
        y = self.encoder(*y)
        return self.kernel(x, y)

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
        # covariance computation
        return self.kernel(x, y)
