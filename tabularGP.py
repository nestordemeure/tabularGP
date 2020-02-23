# Tabular GP
# Gaussian process based tabular model
# source: TODO

import gpytorch
from fastai.torch_core import *
from fastai.tabular import *

__all__ = ['tabularGP_learner']

#------------------------------------------------------------------------------
# GP specific wrappers

# TODO make this code more efficient
def _tensors_of_dl(dl):
    "Takes a dataloader and returns all of its content converted to a ((cat,cont),label) tuple."
    cat = []
    cont = []
    labels = []
    for x,y in iter(dl):
        cat.append(x[0])
        cont.append(x[1])
        labels.append(y)
    cat = torch.cat(cat)
    cont = torch.cat(cont)
    labels = torch.cat(labels)
    return (cat,cont), labels

def _metrics_wrapper(metrics):
    "wraps all metrics so that they can take a multivariate normal as input"
    metrics = [(lambda mInput: metric(mInput[0].mean, mInput[1])) for metric in listify(metrics)]
    return metrics

def _gp_loss(likelihood, model):
    "takes a likelihood, a model and builds an appropriate loss function to train a gaussian process"
    # TODO this is specific to exactGP and gaussian likelyhood
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # TODO we should encourage very large batches
    loss_func = lambda lossInput: -marginal_log_likelihood(lossInput[0], lossInput[1]) # output, target
    return loss_func

#------------------------------------------------------------------------------
# Model

#TabularModel
class TabularGPModel(gpytorch.models.ExactGP):
    "Gaussian process based model for tabular data."
    def __init__(self, nb_continuous_inputs:int, embedding_sizes:ListSizes, output_size:int, 
                 train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # input/outputs parameters
        self.nb_cont = nb_continuous_inputs
        self.nb_cat = len(embedding_sizes)
        self.out_size = output_size
        self.embedding_sizes = embedding_sizes # ListSizes:nb elements in cat, size of embedding
        if self.out_size != 1: raise Exception("Error: this model is not compatible with multioutput problems yet!")
        # gp definition
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.cat_covars = [gpytorch.kernels.IndexKernel(nb_cat, embeding_size) for nb_cat,embeding_size in self.embedding_sizes]
        self.cont_covars = [gpytorch.kernels.RBFKernel() for _ in range(self.nb_cont)]

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        # computes mean
        mean_x = self.mean_module((x_cat, x_cont))
        # computes covariances for individual dimensions
        x = []
        for i,cov in enumerate(self.cat_covars): x.append(cov(x_cat[:,i]))
        for i,cov in enumerate(self.cont_covars): x.append(cov(x_cont[:,i]))
        print(type(x[0]))
        print(x[0].shape)
        x = torch.cat(x, 1)
        # fuse covariances
        covar_x = x[0] # TODO
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#tabular_learner
def tabularGP_learner(data:DataBunch, embedding_sizes:Dict[str,int]=None, metrics=None, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the given data"
    train_x, train_y = _tensors_of_dl(data.train_dl)
    # TODO use likelyhood appropriate for a given task (regresison, multyoutput regression, classification)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    embedding_sizes = data.get_emb_szs(ifnone(embedding_sizes, {}))
    model = TabularGPModel(nb_continuous_inputs=len(data.cont_names), embedding_sizes=embedding_sizes, output_size=data.c,
                           train_x=train_x, train_y=train_y, likelihood=likelihood)
    # finding optimal model hyper parameters
    print("finding optimal model hyper parameters...")
    model.train()
    likelihood.train()
    return Learner(data, model, metrics=_metrics_wrapper(metrics), loss_func=_gp_loss(likelihood,model), **learn_kwargs)

#------------------------------------------------------------------------------
# Test

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')

# problem definition
dep_var = 'age'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'salary']
cont_names = ['fnlwgt', 'education-num']
procs = [FillMissing, Normalize, Categorify]

# load data
dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_idx(list(range(800,1000)))
                  .label_from_df(cols=dep_var, label_cls=FloatList)
                  .databunch())

# classical model
learn = tabular_learner(dls, layers=[200,100], metrics=mae)
learn.fit(1, 1e-2)

# gp model
glearn = tabularGP_learner(dls, layers=[200,100], metrics=mae)
glearn.fit(1, 1e-2)
