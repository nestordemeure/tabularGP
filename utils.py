import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['psd_safe_cholesky', 'recycling_cholesky',
          'soft_clamp_max', 'soft_clamp', 'magnitude', 'magnitude_reciprocal',
          'Scale', 'freeze', 'unfreeze']

#--------------------------------------------------------------------------------------------------
# Mathematical functions

def soft_clamp_max(x, max_value):
    "clamp_max but differentiable"
    return x - F.softplus(x - max_value)

def soft_clamp(x, min_value, max_value):
    """insures that the output is in the given interval
    NOTE: this is not really a clamp"""
    return min_value + (max_value - min_value)*F.sigmoid(x)

def magnitude(x):
    "similar to log but defined on all numbers and returns the magnitude of the input"
    return torch.sign(x) * torch.log1p(torch.abs(x))

def magnitude_reciprocal(x):
    "reciprocal of magnitude function"
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

#--------------------------------------------------------------------------------------------------
# Modules

class Scale(nn.Module):
    "scales the input with positive weights"
    def __init__(self, nb_features:int):
        super().__init__()
        self.sqrt_scale = nn.Parameter(torch.ones(nb_features))

    def forward(self, x):
        scale = self.sqrt_scale * self.sqrt_scale
        return scale * x

def freeze(model):
    "freezes a model"
    for parameter in model.parameters():
        parameter.requires_grad = False

def unfreeze(model):
    "unfreezes a model"
    for parameter in model.parameters():
        parameter.requires_grad = True

#--------------------------------------------------------------------------------------------------
# Linear algebra

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    Slightly adapted from [gpytorch/utils/cholesky.py](https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/cholesky.py)
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        if jitter is None: jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                return L
            except RuntimeError:
                continue
        raise e

# TODO we might build fast explicit backward function
# TODO there is a loss of matter with the current implementation
def warm_approximate_cholesky_old(A, L, upper=False):
    "suppose that upper is False"
    nb_iter = 3
    alpha = 1.0 - 0.5**(1/nb_iter)
    for i in range(nb_iter):
        # solve L*X = A, we want X=transpose(L)
        L_transp, _ = torch.triangular_solve(A, L, upper=upper)
        #L_transp = torch.lerp(L_transp, L_transp.t(), 0.5)
        if torch.any(torch.isnan(L_transp)):
            print("warm failed", i)
            return psd_safe_cholesky(A, upper=upper)
        else: L = torch.lerp(L, L_transp, alpha)
    return L

def _compute_residual(A, L, upper=False):
    "computes the mean square error of a cholesky decomposition"
    A2 = torch.mm(L.t(),L) if upper else torch.mm(L,L.t())
    return ((A - A2)**2).mean().item()

# def recycling_cholesky(A, L=None, tol=10.0, force_computation=False, upper=False):
#     """
#     performs the cholesky decomposition only if its MSE is larger than 1+tol time the MSE when we last truly did a cholesky decomposition
#     the idea is that if A only barely changed then the previous decomposition is still accurate enough
#     """
#     original_residual = L.previous_residual if not L is None else np.nan
#     threshold = original_residual * (1.0 + tol)
#     new_residual = _compute_residual(A, L, upper) if not L is None else np.nan
#     if force_computation or (L is None) or (new_residual > threshold):
#         # refresh cholesky decomposition
#         L = psd_safe_cholesky(A, upper=upper)#.detach()
#         L.previous_residual = _compute_residual(A, L, upper)
#         L.is_recycled = False
#     else:
#         # recycle cholesky decomposition
#         L = L.detach()
#         L.previous_residual = original_residual
#         L.is_recycled = True
#     return L

def recycling_cholesky(A, L=None, tol=10.0, force_computation=False, upper=False):
    """
    performs the cholesky decomposition only if its MSE is larger than tol time the MSE when we last truly did a cholesky decomposition
    the idea is that if A only barely changed then the previous decomposition is still accurate enough
    """
    assert tol >= 1.0
    if force_computation or (L is None) or (_compute_residual(A, L, upper) > L.previous_residual * tol):
        # refresh cholesky decomposition
        L = psd_safe_cholesky(A, upper=upper, out=L).detach() # we drop the gradient for the cholesky decomposition
        L.previous_residual = _compute_residual(A, L, upper)
        L.is_recycled = False
    else:
        # recycle cholesky decomposition
        L.is_recycled = True
    return L
