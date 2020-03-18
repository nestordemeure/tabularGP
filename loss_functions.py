# Loss functions
# loss functions for probabilistic models
# source: https://github.com/nestordemeure/tabularGP/blob/master/loss_function.py

import numpy as np
import torch
from torch import Tensor

__all__ = ['log_standard_normal_cdf', 'gp_gaussian_marginal_log_likelihood', 'gp_is_greater_log_likelihood']

def log_standard_normal_cdf(x):
    """
    more numerically stable way to get the logarithm of the standard normal cdf
    we approximate the cdf with: 1 / (1 + exp(-k*x)) with k = log(2) * sqrt(2*pi)
    see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
    and use softplus (`softplus(x) = log(1 + exp(x))`) as a way to get a numerically stable implementation
    """
    # erfc(x) = 1 - efr(x)
    # erfc(x) = 1 - (e - e-) / (e + e-) = 2*e- / (e + e-) = 2 / (e2 + 1)
    # erfc(x) = 2 / (1 + exp(2*k*x)) with k = sqrt(pi)*ln(2) (aproximation)
    # cdf(x) = 0.5 * erfc(-x/sqrt(2))
    # cdf(x) = 0.5 * erfc(-x/sqrt(2)) = 1 / (1 + exp(-sqrt(2)*k*x))
    # cdf(x) = 1 / (1 + exp(-ln(2)lsqrt(2*pi)*x))
    # logcdf(x) = -softplus(-ln(2)lsqrt(2*pi)*x)
    k = np.log(2) * np.sqrt(2*np.pi)
    return -torch.nn.functional.softplus(-k*x)

def gp_gaussian_marginal_log_likelihood(prediction, target:Tensor, reduction='mean'):
    "loss function for a regression gaussian process"
    if target.dim() == 1: target = target.unsqueeze(-1)
    mean = prediction
    stdev = prediction.stdev
    minus_log_likelihood = (mean - target)**2 / (2*stdev*stdev) + torch.log(stdev * np.sqrt(2.*np.pi))
    if reduction == 'mean': return minus_log_likelihood.mean()
    elif reduction == 'sum': return minus_log_likelihood.sum()
    else: return minus_log_likelihood

def gp_is_greater_log_likelihood(prediction, target:Tensor, reduction='mean'):
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
    if reduction == 'mean': return minus_log_proba.mean()
    elif reduction == 'sum': return minus_log_proba.sum()
    else: return minus_log_proba
