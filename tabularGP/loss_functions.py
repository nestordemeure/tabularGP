# Loss functions
# loss functions for probabilistic models
# source: https://github.com/nestordemeure/tabularGP/blob/master/loss_function.py

import numpy as np
import torch
from torch import Tensor
from tabularGP.utils import listify

__all__ = ['log_standard_normal_cdf', 'gp_gaussian_marginal_log_likelihood',
           'gp_is_greater_log_likelihood', 'gp_softmax', 'gp_metric_wrapper']

def gp_metric_wrapper(metrics):
    "takes the metrics and wrap them to strip the inputs of their std dimension"
    def wrap_metric(metric):
        def wrapped_metric(prediction, target): return metric(prediction[...,0], target)
        # preserves the name of the metric for display purposes
        wrapped_metric.__name__ = metric.__name__  if hasattr(metric, '__name__') else metric.name
        return wrapped_metric
    return list(map(wrap_metric, listify(metrics)))

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
    mean = prediction[..., 0]
    stdev = prediction[..., 1]
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
    mean = prediction[..., 0]
    stdev = prediction[..., 1]
    # gets the target
    target = target.long() # converts from int to a proper index type
    mean_target = mean[target]
    std_target = stdev[target]
    # computes the probability that the target is larger than another output
    mean_sub = mean_target - mean
    std_sub = torch.sqrt(std_target*std_target + stdev*stdev + 2.0*std_target*stdev)
    # TODO check the formula for log_standard_normal_cdf
    minus_log_proba = -log_standard_normal_cdf(mean_sub / std_sub)
    # removes the probability between the target and itself
    minus_log_proba[target] = 0.0
    # TODO using a sum here is equivalent to doing the product of the proba but they are not independant so its a bit approximativ
    # TODO we could use the expected best of several gaussian ?
    minus_log_proba = torch.sum(minus_log_proba, dim=1)
    if reduction == 'mean': return minus_log_proba.mean()
    elif reduction == 'sum': return minus_log_proba.sum()
    else: return minus_log_proba

def gp_softmax(prediction):
    "takes raw logits, with an additional stdev field, and produces the probability that each class has a larger score than the others"
    standard_normal_cdf = torch.distributions.Normal(Tensor([0.0]).to(prediction.device), Tensor([1.0]).to(prediction.device)).cdf
    # gets the output distributions
    mean = prediction[..., 0]
    stdev = prediction[..., 1]
    # we want to compute the substract between all pairs of mean and std
    if prediction.dim() < 3:
        mean.unsqueeze_(0)
        stdev.unsqueeze_(0)
    mean_target = mean.unsqueeze(-1).expand(-1, -1, mean.size(-1))
    mean = mean.unsqueeze(-2).expand(-1, mean.size(-1), -1)
    std_target = stdev.unsqueeze(-1).expand(-1, -1, stdev.size(-1))
    stdev = stdev.unsqueeze(-2).expand(-1, stdev.size(-1), -1)
    # computes the probability that each class is larger than the others
    mean_sub = mean_target - mean
    std_sub = torch.sqrt(std_target*std_target + stdev*stdev + 2.0*std_target*stdev)
    proba = standard_normal_cdf(mean_sub / std_sub)
    # time 2 to compense the 0.5 probability of a class being greater than itself
    return torch.prod(proba, dim=-1) * 2.0
