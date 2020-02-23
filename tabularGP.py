# Tabular GP
# Gaussian process based tabular model
# source: TODO

from gpytorch import *
from fastai.torch_core import *
from fastai.tabular import *

__all__ = ['tabularGP_learner']

#------------------------------------------------------------------------------
# Encoders

#------------------------------------------------------------------------------
# Model

#TabularModel
class TabularGPModel(Module):
    "Gaussian process based model for tabular data."
    def __init__(self, nb_continuous_inputs:int, nb_categorial_inputs:int, output_size:int, encoder=None):
        super().__init__()
        self.encoder = encoder
        self.nb_cont = nb_continuous_inputs
        self.nb_cat = nb_categorial_inputs
        self.out_size = output_size
        if self.out_size != 1: raise Exception("Error: this model is not compatible with multioutput problems yet!")

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return x

#tabular_learner
def tabularGP_learner(data:DataBunch, encoder=None, metrics=None, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the given data"
    model = TabularGPModel(nb_continuous_inputs=len(data.cont_names), nb_categorial_inputs=len(data.cat_names), output_size=data.c,
                           encoder=encoder)
    return Learner(data, model, metrics=metrics, **learn_kwargs)

#------------------------------------------------------------------------------
# Test

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')

# problem definition
dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [FillMissing, Normalize, Categorify]

# load data
dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_idx(list(range(800,1000)))
                  .label_from_df(cols=dep_var)
                  .databunch())

# classical model
learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2)
