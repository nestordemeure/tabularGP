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
    # TODO use kmean clustering
    dataset = dataset[:nb_inducing_points, :]
    return dataset

#------------------------------------------------------------------------------
# Model

#TabularModel
class TabularGPModel(gpytorch.models.ApproximateGP):
    "Gaussian process based model for tabular data."
    def __init__(self, nb_continuous_inputs:int, embedding_sizes:ListSizes, output_size:int, inducing_points, likelihood):
        # defines variational strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        if output_size > 1:
            variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(variational_strategy, num_tasks=output_size)
        # stores base members
        super().__init__(variational_strategy)
        self.likelihood = likelihood
        self.output_size = output_size
        self.nb_continuous_inputs = nb_continuous_inputs
        self.register_buffer("category_sizes", torch.LongTensor([nb_cat for nb_cat,embeding_size in embedding_sizes]))
        # defines mean
        self.mean_module = gpytorch.means.ConstantMean()
        if output_size > 1: self.mean_module = gpytorch.means.MultitaskMean(self.mean_module, num_tasks=output_size)
        # defines covariance kernels
        self.cat_covars = nn.ModuleList([ScaleKernel(IndexKernel(nb_cat, embeding_size)) for nb_cat,embeding_size in embedding_sizes])
        self.cont_covars = nn.ModuleList([ScaleKernel(RBFKernel()) for _ in range(nb_continuous_inputs)])
        # taken from MultitaskKernel
        if output_size > 1:
            self.task_covar_module = IndexKernel(num_tasks=output_size)

    #def forward(self, x_cat:Tensor, x_cont:Tensor):
    def forward(self, inputs:Tensor):
        # gets the input back into a usable form
        # TODO we could encapsulate that into a dedicated function
        x_cat = inputs[:, self.nb_continuous_inputs:]
        x_cont = inputs[:, :self.nb_continuous_inputs]
        # converts x_cat to indexes between 0 and their maximum allowed value
        x_cat = torch.min(x_cat.long().clamp(min=0), self.category_sizes.unsqueeze(0)-1)
        # computes covariances
        # TODO use better kernel
        cat_covars = enumerate(self.cat_covars)
        i,cov = next(cat_covars)
        covar_x = cov(x_cat[:,i])
        for i,cov in cat_covars: covar_x += cov(x_cat[:,i])
        for i,cov in enumerate(self.cont_covars): covar_x += cov(x_cont[:,i])
        # computes mean
        mean_x = self.mean_module(inputs)
        # returns a distribution
        if self.output_size == 1:
            return MultivariateNormal(mean_x, covar_x)
        else:
            # taken from MultitaskKernel
            covar_i = self.task_covar_module.covar_matrix
            covar_x = gpytorch.lazy.KroneckerProductLazyTensor(covar_x, covar_i)
            return MultitaskMultivariateNormal(mean_x, covar_x)

    def __call__(self, x_cat:Tensor, x_cont:Tensor, **kwargs):
        # use an inputs format that is compatible with the variational strategy implementation (single float tensor)
        inputs = torch.cat((x_cat.float(), x_cont), dim=1)
        return self.variational_strategy(inputs)

#tabular_learner
def tabularGP_learner(data:DataBunch, nb_inducing_points = 500, embedding_sizes:Dict[str,int]=None, metrics=None, **learn_kwargs):
    "Builds a `TabularGPModel` model and outputs a `Learner` that encapsulate the model and the given data"
    # picks a likelihood for the task
    is_classification = hasattr(data, 'classes')
    is_multitask = data.c > 1
    if is_classification: likelihood = SoftmaxLikelihood(num_classes=data.c, num_features=data.c)
    elif is_multitask: likelihood = MultitaskGaussianLikelihood(data.c)
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
