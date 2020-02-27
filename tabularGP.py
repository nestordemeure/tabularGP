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

def _gp_loss(likelihood, model, num_data):
    "takes a likelihood, a model and builds an appropriate loss function to train a gaussian process"
    marginal_log_likelihood = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data)
    def loss_func(output,target): return -marginal_log_likelihood(output, target)
    return loss_func

def _defines_inducing_points(data:DataBunch, nb_inducing_points:int):
    # extracts all the dataset as a single tensor
    dataset = []
    for x,_ in iter(data.train_dl):
        xcat = x[0].float()
        xcont = x[1]
        x = torch.cat((xcat,xcont), dim=1)
        dataset.append(x)
    # concat the batches
    dataset = torch.cat(dataset)
    # selects inducing points
    # TODO use kmean clustering to find representative inducing points
    dataset = dataset[:nb_inducing_points, :]
    return dataset

def safe_MultivariateNormal(mean_x, covar_x):
    # evaluate the kernel to avoid a bug
    # while computing diagonal of lazy tensor
    # when the kernel contains a sum of IndexKernels
    # TODO reproduce sum of IndexKernel bug without fastai and forward it to gpytorch authors
    covar_x = covar_x.evaluate()

    # inspired by psd_safe_cholesky
    # https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/cholesky.py
    try:
        # does the transformation works with no diagonal noise ?
        multivariateNormal = MultivariateNormal(mean_x, covar_x)
        return multivariateNormal
    except RuntimeError as e:
        # tries to succed with incremental level of noise
        jitter = 1e-6 if mean_x.dtype == torch.float32 else 1e-8
        jitter_prev = 0
        for i in range(6):
            jitter_new = jitter * (10 ** i)
            covar_x.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                # doe sit work with this level of noise ?
                multivariateNormal = MultivariateNormal(mean_x, covar_x)
                warnings.warn(f"Covariance matrix not p.d., added jitter of {jitter_new} to the diagonal", RuntimeWarning)
                return multivariateNormal
            except RuntimeError: continue
        # did not work even with large levels of noise, we give up
        raise e

#------------------------------------------------------------------------------
# Model

class TabularGPModel(gpytorch.models.ApproximateGP):
    "Gaussian process based model for tabular data."
    def __init__(self, nb_continuous_inputs:int, embedding_sizes:ListSizes, output_size:int, inducing_points, likelihood):
        # repeats for multi output case
        nb_inducing_points = inducing_points.size(-2)
        if output_size > 1:
            # TODO expand might reduce memory usage but how does it impact precision ?
            inducing_points = inducing_points.unsqueeze(dim=0).repeat(output_size,1,1)
        # defines variational strategy
        batch_shape = torch.Size() if output_size == 1 else torch.Size([output_size])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(nb_inducing_points, batch_shape=batch_shape)
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        if output_size > 1: variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(variational_strategy, num_tasks=output_size)
        # stores base members
        super().__init__(variational_strategy)
        self.likelihood = likelihood
        self.output_size = output_size
        self.nb_continuous_inputs = nb_continuous_inputs
        self.register_buffer("category_sizes", torch.LongTensor([nb_cat for nb_cat,embeding_size in embedding_sizes]))
        # defines mean
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        #if output_size > 1: self.mean_module = gpytorch.means.MultitaskMean(self.mean_module, num_tasks=output_size)
        # defines covariance kernels
        self.cat_covars = nn.ModuleList([IndexKernel(num_tasks=nb_cat, rank=embedding_size, batch_shape=batch_shape) for nb_cat,embedding_size in embedding_sizes])
        #self.cont_covars = nn.ModuleList([ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape) for _ in range(nb_continuous_inputs)])
        self.cont_covar = AdditiveStructureKernel(ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape), num_dims=nb_continuous_inputs)

    #def forward(self, x_cat:Tensor, x_cont:Tensor):
    def forward(self, inputs:Tensor):
        # gets the input back into a category/continuous form
        x_cat = inputs[..., :-self.nb_continuous_inputs].long()
        x_cont = inputs[..., -self.nb_continuous_inputs:]
        # insures that x_cat is between 0 and the maximum allowed value
        #x_cat = torch.min(x_cat.clamp_min(0), self.category_sizes-1)
        # computes covariances
        # TODO use better kernel
        covar_x = self.cont_covar(x_cont)
        for i,cov in enumerate(self.cat_covars): covar_x += cov(x_cat[...,i].t())
        # computes mean
        mean_x = self.mean_module(inputs)
        # returns a distribution
        return safe_MultivariateNormal(mean_x, covar_x)

    def __call__(self, x_cat:Tensor, x_cont:Tensor, **kwargs):
        # use an inputs format that is compatible with the variational strategy implementation (single float tensor)
        inputs = torch.cat((x_cat.float(), x_cont), dim=1)
        return self.variational_strategy(inputs)

def tabularGP_learner(data:DataBunch, nb_inducing_points = 500, embedding_sizes:Dict[str,int]=None, metrics=None, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the given data"
    # picks a likelihood for the task
    is_classification = hasattr(data, 'classes')
    is_multitask = data.c > 1
    if is_classification: likelihood = SoftmaxLikelihood(num_classes=data.c, num_features=data.c)
    elif is_multitask: likelihood = MultitaskGaussianLikelihood(num_tasks=data.c)
    else: likelihood = GaussianLikelihood()
    # gets information from the dataset
    inducing_points = _defines_inducing_points(data, nb_inducing_points)
    dataset_size = len(data.train_dl.dl.dataset)
    # defines the model
    embedding_sizes = data.get_emb_szs(ifnone(embedding_sizes, {}))
    model = TabularGPModel(nb_continuous_inputs=len(data.cont_names), embedding_sizes=embedding_sizes,
                           output_size=data.c, inducing_points=inducing_points, likelihood=likelihood)
    # finds optimal model hyper parameters
    model.train()
    likelihood.train()
    return Learner(data, model, metrics=_metrics_wrapper(metrics), loss_func=_gp_loss(likelihood,model,num_data=dataset_size), **learn_kwargs)

#------------------------------------------------------------------------------
# Regression

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv').sample(1000)

# problem definition
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['education-num', 'fnlwgt']
procs = [FillMissing, Normalize, Categorify]

# load data
dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_rand_pct()
                  .label_from_df(cols='age', label_cls=FloatList)
                  .databunch(bs=64))

# classical model
#learn = tabular_learner(dls, layers=[200,100], metrics=[rmse, mae])
#learn.fit(10, 1e-2)

# gp model
glearn = tabularGP_learner(dls, metrics=[rmse, mae], nb_inducing_points=500)
glearn.fit(10, 1e-2)

#------------------------------------------------------------------------------
# Multiple regression

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv').sample(1000)

# problem definition
cont_names = ['education-num']

# load data
dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_rand_pct()
                  .label_from_df(cols=['age','fnlwgt'], label_cls=FloatList)
                  .databunch())

# gp model
glearn = tabularGP_learner(dls, metrics=[rmse, mae])
glearn.fit(10, 0.1)

#------------------------------------------------------------------------------
# Classification

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['education-num', 'fnlwgt', 'age']

dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_rand_pct()
                  .label_from_df(cols='salary')
                  .databunch())

# gp model
glearn = tabularGP_learner(dls, metrics=accuracy)
glearn.fit(10, 1e-1)

#learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
#learn.fit(10, 1e-2)