import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['psd_safe_cholesky', 'Scale', 'soft_clamp_max', 'soft_clamp']

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
        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                #warnings.warn(f"A not p.d., added jitter of {jitter_new} to the diagonal", RuntimeWarning)
                return L
            except RuntimeError:
                continue
        raise e

class Scale(nn.Module):
    "scales the input with positive weights"
    def __init__(self, nb_features:int):
        super().__init__()
        self.sqrt_scale = nn.Parameter(torch.ones(nb_features))

    def forward(self, x):
        scale = self.sqrt_scale * self.sqrt_scale
        return scale * x

def soft_clamp_max(x, max_value):
    "clamp_max but differentiable"
    return x - F.softplus(x - max_value)

def soft_clamp(x, min_value, max_value):
    """insures that the output is in the given interval
    NOTE: this is not really a clamp"""
    return min_value + (max_value - min_value)*F.sigmoid(x)
