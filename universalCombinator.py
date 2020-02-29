# Universal combinators
# This work can be seen as a prolongation of "A Logarithmic Neural Network Architecture for Unbounded Non-Linear Function Approximation" [0]
# that adresses its shortcomings (how to deal with negative inputs) and give examples of architectures making use of the concept (polynomial and product of sums).
# [0] https://www.researchgate.net/publication/2643819_A_Logarithmic_Neural_Network_Architecture_for_Unbounded_Non-Linear_Function_Approximation
# source: https://github.com/nestordemeure/tabularGP/blob/master/universalCombinator.py

import numpy as np
from torch.nn import Linear, Module, Parameter
import torch.nn.functional as F
import torch
# my imports
from utils import soft_clamp_max

__all__ = ['PositiveLinear', 'PositiveMultiply', 'Multiply', 'Polynomial', 'PositiveProductOfSum']

#--------------------------------------------------------------------------------------------------
# Building Blocks

class PositiveLinear(Linear):
    "Similar to the Linear module but insures that the weights are positiv (thus positive inputs will stay positive)"
    __constants__ = ['bias', 'in_features', 'out_features', 'use_exponential']

    def __init__(self, in_features, out_features, bias=True, use_exponential=False):
        "by default, positivity is enforced with a square, if you only care about the magnitude of the parameter, set `use_exponential` to true to use an exponential instead"
        super().__init__(in_features, out_features, bias)
        self.use_exponential = use_exponential

    def forward(self, input):
        "uses either an exponential or a square to insure that weights are positive"
        if self.use_exponential:
            weight = torch.exp(self.weight)
            bias = self.bias if self.bias is None else torch.exp(self.bias)
        else:
            weight = self.weight * self.weight
            bias = self.bias if self.bias is None else self.bias * self.bias
        return F.linear(input, weight, bias)

# Introduces a block that can learn the product of its inputs raised to arbitrary powers.
# We provide a positive version and a general version that can also take negative inputs. 
# When possible the positive version is recommended as it is faster and trains better (no discontinuity along negative inputs).

class PositiveMultiply(Module):
    """Learns one products of arbitrary power of inputs per output.
    WARNING: this module assumes that its inputs will be positive (and insures that the output will stay positive)"""
    __constants__ = ['bias', 'in_features', 'out_features', 'epsilon']

    def __init__(self, in_features, out_features, bias=True, epsilon=1e-8, maximum_output=1e7):
        """`epsilon` is there to insure that there will be no 0 in the inputs (causing Nan or infinities in the outputs)
        adding a bias as the effect of multiplying the output with a constant"""
        super().__init__()
        self.register_buffer('epsilon', torch.Tensor([epsilon]))
        self.in_features = in_features
        self.out_features = out_features
        self.log_maximum_output = np.log(maximum_output)
        # weights initialized to start equivalent to a geometric mean
        self.weight = Parameter(torch.ones(out_features, in_features) / in_features)
        if bias: self.bias = Parameter(torch.zeros(out_features))
        else: self.register_parameter('bias', None)

    def forward(self, input):
        "applies a linear operation in log space"
        log_input = torch.log(input + self.epsilon)
        log_output = F.linear(log_input, self.weight, self.bias)
        log_output = soft_clamp_max(log_output, self.log_maximum_output) # insures it does not degrate into inf
        return torch.exp(log_output)

class Multiply(PositiveMultiply):
    """Learns one products of arbitrary power of inputs per output."""

    def forward(self, input):
        "applies a linear operation in log space using Euler's formula to take the complex logarithm of negative inputs"
        # takes the logarithm of the input
        input_real = torch.log(torch.abs(input) + self.epsilon)
        # uses euler's formula to get the imaginary part for the complex log
        input_imaginary = torch.zeros(input.shape)
        input_imaginary[input_imaginary < 0.0] = np.pi
        # applies the linear transformation in log space
        output_real = F.linear(input_real, self.weight, self.bias)
        output_real = soft_clamp_max(output_real, self.log_maximum_output) # insures it does not degrade into inf
        output_imaginary = F.linear(input_imaginary, self.weight, self.bias)
        # gets out of log space (note that there is no sinus as we inforce a real output)
        return torch.exp(output_real) * torch.cos(output_imaginary)

#--------------------------------------------------------------------------------------------------
# Combinators

# A module that can learn any polynomial with k terms (and not simply a polynomial of order k):

class Polynomial(Module):
    """Learns any k terms polynomial"""
    __constants__ = ['in_features', 'out_features', 'nb_terms']

    def __init__(self, in_features, out_features, nb_terms, epsilon=1e-8, maximum_term_value=1e7):
        "`nb_terms` is the number of terms of the polynomial and not its order"
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nb_terms = nb_terms
        self.product = Multiply(in_features, nb_terms, bias=False, epsilon=epsilon, maximum_output=maximum_term_value)
        self.addition = Linear(nb_terms, out_features, bias=True)

    def forward(self, input):
        "takes the weighted sum of the polynomial terms"
        return self.addition(self.product(input))

# Any product of weighted sums of inputs. This is useful to combine similarity scores (ie for gaussian process).

class PositiveProductOfSum(Module):
    """Learns any product of sum of the inputs.
    WARNING: works under the hypothesis that the inputs are positive"""
    __constants__ = ['in_features', 'out_features', 'nb_sum']

    def __init__(self, in_features, out_features, nb_sums=None, epsilon=1e-8, maximum_output=1e7):
        "`nb_sums` is `in_features` by default it corresponds to the number of sums that will be multiplied together"
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nb_sums = in_features if nb_sums is None else nb_sums
        self.addition = PositiveLinear(in_features, self.nb_sums, bias=False, use_exponential=False)
        self.product = PositiveMultiply(self.nb_sums, out_features, bias=True, epsilon=epsilon, maximum_output=maximum_output)

    def forward(self, input):
        "takes the product of the weighted sum of the inputs"
        res = self.product(self.addition(input))
        return res

