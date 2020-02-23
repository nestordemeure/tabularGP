# Tabular GP
# Gaussian process based tabular model
# source: https://github.com/nestordemeure/tabularGP/blob/master/tabularGP.py

import gpytorch
from fastai.torch_core import *
from fastai.tabular import *

__all__ = ['TabularGPModel', 'tabularGP_learner']

#------------------------------------------------------------------------------
# GP specific wrappers

def _metrics_wrapper(metrics):
    "wraps all provided metrics so that they can take a multivariate normal as input"
    # TODO this rename the metric which is not an expected behaviour
    def apply_metric_to_mean(output, target, metric=None): return metric(output.mean, target)
    metrics = [partial(apply_metric_to_mean, metric=m) for m in listify(metrics)]
    return metrics

def _gp_loss(likelihood, model):
    "takes a likelihood, a model and builds an appropriate loss function to train a gaussian process"
    # TODO this is specific to exactGP and gaussian likelihood
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    def loss_func(output,target): return -marginal_log_likelihood(output, target)
    return loss_func

#------------------------------------------------------------------------------
# Model

#TabularModel
class TabularGPModel(gpytorch.models.ExactGP):
    "Gaussian process based model for tabular data."
    def __init__(self, nb_continuous_inputs:int, embedding_sizes:ListSizes, output_size:int, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        if output_size != 1: raise Exception("Error: this model is not compatible with multioutput problems yet!")
        self.mean_module = gpytorch.means.ConstantMean()
        # we have one kernel per input column
        self.cat_covars = nn.ModuleList([gpytorch.kernels.IndexKernel(nb_cat, embeding_size) for nb_cat,embeding_size in embedding_sizes])
        self.cont_covars = nn.ModuleList([gpytorch.kernels.RBFKernel() for _ in range(nb_continuous_inputs)])

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        # computes mean
        mean_x = self.mean_module(x_cont) # TODO here we ignore x_cat
        # computes covariances for individual dimensions
        x = []
        for i,cov in enumerate(self.cat_covars): x.append(cov(x_cat[:,i]))
        for i,cov in enumerate(self.cont_covars): x.append(cov(x_cont[:,i]))
        covar_x = x[0] # TODO fuse inputs (which are covariance matrix!)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#tabular_learner
def tabularGP_learner(data:DataBunch, embedding_sizes:Dict[str,int]=None, metrics=None, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the given data"
    # insures training will be done on the full dataset and not smaller batches
    data.train_dl.batch_size = len(data.train_dl.dl.dataset)
    data.train_dl = data.train_dl.new(shuffle=False)
    train_x, train_y = next(iter(data.train_dl))
    # TODO use likelyhood appropriate for a given task (regression, multioutput regression, classification)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    embedding_sizes = data.get_emb_szs(ifnone(embedding_sizes, {}))
    model = TabularGPModel(nb_continuous_inputs=len(data.cont_names), embedding_sizes=embedding_sizes, output_size=data.c,
                           train_x=train_x, train_y=train_y, likelihood=likelihood)
    # finding optimal model hyper parameters
    model.train()
    likelihood.train()
    return Learner(data, model, metrics=_metrics_wrapper(metrics), loss_func=_gp_loss(likelihood,model), **learn_kwargs)

# TODO gets to scale
# https://github.com/cornellius-gp/gpytorch/tree/master/examples/02_Scalable_Exact_GPs

#------------------------------------------------------------------------------
# Test

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv').sample(1000)

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
#learn = tabular_learner(dls, layers=[200,100], metrics=mae)
#learn.fit(1, 1e-2)

# gp model
glearn = tabularGP_learner(dls, metrics=[rmse, mae])
glearn.fit(10, 1e-2)
