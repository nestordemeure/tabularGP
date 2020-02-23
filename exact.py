# Tabular GP
# Gaussian process based tabular model
# source: https://github.com/nestordemeure/tabularGP/blob/master/tabularGP.py

import gpytorch
from gpytorch.kernels import *
from gpytorch.likelihoods import *
from gpytorch.distributions import *
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
        self.output_size = output_size
        # defines mean
        self.mean_module = gpytorch.means.ConstantMean()
        if output_size > 1: self.mean_module = gpytorch.means.MultitaskMean(self.mean_module, num_tasks=output_size)
        # defines covariance kernels
        self.cat_covars = nn.ModuleList([ScaleKernel(IndexKernel(nb_cat, embeding_size)) for nb_cat,embeding_size in embedding_sizes])
        self.cont_covars = nn.ModuleList([ScaleKernel(RBFKernel()) for _ in range(nb_continuous_inputs)])
        # taken from MultitaskKernel
        #self.multicov = MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=output_size)
        self.task_covar_module = IndexKernel(num_tasks=output_size)

    def forward(self, x_cat:Tensor, x_cont:Tensor):
        # computes covariances
        # TODO use better kernel
        cat_covars = enumerate(self.cat_covars)
        i,cov = next(cat_covars)
        covar_x = cov(x_cat[:,i])
        for i,cov in cat_covars: covar_x += cov(x_cat[:,i])
        for i,cov in enumerate(self.cont_covars): covar_x += cov(x_cont[:,i])
        # computes mean
        mean_x = self.mean_module(x_cont) # TODO here we ignore x_cat
        # returns a distribution
        if self.output_size == 1:
            return MultivariateNormal(mean_x, covar_x)
        else:
            # taken from MultitaskKernel
            covar_i = self.task_covar_module.covar_matrix
            covar_x = gpytorch.lazy.KroneckerProductLazyTensor(covar_x, covar_i)
            return MultitaskMultivariateNormal(mean_x, covar_x)

#tabular_learner
def tabularGP_learner(data:DataBunch, embedding_sizes:Dict[str,int]=None, metrics=None, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the given data"
    # insures training will be done on the full dataset and not smaller batches
    data.train_dl.batch_size = len(data.train_dl.dl.dataset)
    data.train_dl = data.train_dl.new(shuffle=False)
    train_x, train_y = next(iter(data.train_dl))
    # picks a likelihood for the task
    is_classification = hasattr(data, 'classes')
    is_multitask = data.c > 1
    if is_classification: raise Exception("You cannot use exactGP for classification tasks!")
    #if is_classification: likelihood = SoftmaxLikelihood(num_classes=data.c, num_features=data.c)
    elif is_multitask: likelihood = MultitaskGaussianLikelihood(data.c)
    else: likelihood = GaussianLikelihood()
    # defines the model
    embedding_sizes = data.get_emb_szs(ifnone(embedding_sizes, {}))
    model = TabularGPModel(nb_continuous_inputs=len(data.cont_names), embedding_sizes=embedding_sizes,
                           output_size=data.c, train_x=train_x, train_y=train_y, likelihood=likelihood)
    # finds optimal model hyper parameters
    model.train()
    likelihood.train()
    return Learner(data, model, metrics=_metrics_wrapper(metrics), loss_func=_gp_loss(likelihood,model), **learn_kwargs)

#------------------------------------------------------------------------------
# Test

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv').sample(1000)

# problem definition
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['fnlwgt', 'education-num']
procs = [FillMissing, Normalize, Categorify]

# load data
dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_rand_pct()
                  .label_from_df(cols='age', label_cls=FloatList)
                  .databunch())

# classical model
#learn = tabular_learner(dls, layers=[200,100], metrics=[rmse, mae])
#learn.fit(10, 1e-2)

# gp model
glearn = tabularGP_learner(dls, metrics=[rmse, mae])
glearn.fit(10, 0.1)

#------------------------------------------------------------------------------
# Classification

#dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
#                  .split_by_rand_pct()
#                  .label_from_df(cols='salary')
#                  .databunch())

# gp model
#glearn = tabularGP_learner(dls, metrics=accuracy)
#glearn.fit(10, 1e-2)
