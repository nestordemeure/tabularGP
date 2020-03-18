# adapted from: https://github.com/sbarratt/torch_cg
import torch

__all__ = ['cg', 'CG']

def cg(A, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None):
    """Solves a PD matrix linear systems using the preconditioned CG algorithm.

    This function solves a linear systems of the form

        A X = B

    where A is a n x n positive definite matrix and B_i is a n x m matrix,
    and X is the n x m matrix representing the solution for the ith system.

    Args:
        A: A callable that performs a batch matrix multiply of A and a n x m matrix.
        B: A n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=n)
    """
    n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x # TODO a diagonal preconditioner might be a better start
        if X0 is None: X0 = X0 = B / A.diagonal().unsqueeze(-1)
    if X0 is None: X0 = M_bmm(B)
    if maxiter is None: maxiter = n

    assert B.shape == (n, m)
    assert X0.shape == (n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - torch.mm(A,X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=0)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    for k in range(1, maxiter + 1):
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(0)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(0) / denominator
            P_k = Z_k1 + beta.unsqueeze(0) * P_k1

        denominator = (P_k * torch.mm(A,P_k)).sum(0)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(0) / denominator
        X_k = X_k1 + alpha.unsqueeze(0) * P_k
        R_k = R_k1 - alpha.unsqueeze(0) * torch.mm(A,P_k)

        residual_norm = torch.norm(torch.mm(A,X_k) - B, dim=0)

        if (residual_norm <= stopping_matrix).all(): break

    #print(X_k.min(), X_k.max(), X_k.mean())
    return X_k


class CG(torch.autograd.Function):

    def __init__(self, A, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None):
        self.A = A
        self.M_bmm = M_bmm
        self.X0 = X0
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter

    def forward(self, B):
        return cg(self.A, B, M_bmm=self.M_bmm, X0=self.X0, rtol=self.rtol, atol=self.atol, maxiter=self.maxiter)

    def backward(self, dX):
        return cg(self.A, dX, M_bmm=self.M_bmm, rtol=self.rtol, atol=self.atol, maxiter=self.maxiter)
