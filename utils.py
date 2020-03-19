import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['psd_safe_cholesky',
          'soft_clamp_max', 'magnitude', 'magnitude_reciprocal',
          'Scale', 'freeze', 'unfreeze']

#--------------------------------------------------------------------------------------------------
# Mathematical functions

def soft_clamp_max(x, max_value):
    "clamp_max but differentiable"
    return x - F.softplus(x - max_value)

def magnitude(x):
    "similar to log but defined on all numbers and returns the magnitude of the input"
    return torch.sign(x) * torch.log1p(torch.abs(x))

def magnitude_reciprocal(x):
    "reciprocal of magnitude function"
    return torch.sign(x) * torch.expm1(torch.abs(x))

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
