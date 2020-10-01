# Priors
# Priors for gaussian process based models
# a prior is a very simple prediction function which is used in combinaison with the gaussian process
# source: https://github.com/nestordemeure/tabularGP/blob/master/prior.py

import abc
# library imports
from fastai.layers import Embedding
from torch import nn, Tensor
import torch

__all__ = ['ZeroPrior', 'ConstantPrior', 'LinearPrior']

#--------------------------------------------------------------------------------------------------
# abstract class

class Prior(nn.Module):
    "Abstract class for priors."
    def __init__(self, train_input_cat:Tensor, train_input_cont:Tensor, train_outputs:Tensor, embedding_sizes):
        "Note that a pretrained prior does not need to implement this exact same constructor as only the forward will be called."
        super().__init__()

    @abc.abstractmethod
    def forward(self, x_cat:Tensor, x_cont:Tensor):
        "Makes a prediction with the given inputs."

#--------------------------------------------------------------------------------------------------
# priors

class ZeroPrior(Prior):
    "Prior that ignores its inputs and returns zero"
    def __init__(self, train_input_cat:Tensor, train_input_cont:Tensor, train_outputs:Tensor, embedding_sizes):
        super().__init__(train_input_cat, train_input_cont, train_outputs, embedding_sizes)
        nb_outputs = train_outputs.size(-1)
        self.register_buffer('output', torch.zeros(nb_outputs))

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        return self.output

class ConstantPrior(Prior):
    "Prior that ignores its inputs and returns a constant"
    def __init__(self, train_input_cat:Tensor, train_input_cont:Tensor, train_outputs:Tensor, embedding_sizes):
        super().__init__(train_input_cat, train_input_cont, train_outputs, embedding_sizes)
        self.output = nn.Parameter(train_outputs.mean(dim=0))

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        return self.output

class LinearPrior(Prior):
    "Prior that fits a linear model on the inputs"
    def __init__(self, train_input_cat:Tensor, train_input_cont:Tensor, train_outputs:Tensor, embedding_sizes):
        super().__init__(train_input_cat, train_input_cont, train_outputs, embedding_sizes)
        self.embeddings = nn.ModuleList([Embedding(ni, nf) for ni,nf in embedding_sizes])
        self.nb_embeddings = sum(e.embedding_dim for e in self.embeddings)
        self.nb_cont = train_input_cont.size(-1)
        nb_outputs = train_outputs.size(-1)
        self.model = nn.Linear(in_features=self.nb_embeddings+self.nb_cont, out_features=nb_outputs)

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        if self.nb_embeddings != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
            x = torch.cat(x, 1)
        if self.nb_cont != 0:
            x = torch.cat([x, x_cont], 1) if self.nb_embeddings != 0 else x_cont
        return self.model(x)
