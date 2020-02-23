# non fastai based loop to confirm that OOM are normal and expected
import gpytorch
from gpytorch.kernels import *
from gpytorch.likelihoods import *
from gpytorch.distributions import *
from fastai.torch_core import *
from fastai.tabular import *

# model
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
        #self.cont_covars = AdditiveStructureKernel(ScaleKernel(RBFKernel()), num_dims=nb_continuous_inputs)
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

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv').sample(1000)
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['fnlwgt', 'education-num']
procs = [FillMissing, Normalize, Categorify]
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_idx(list(range(800,1000)))
                  .label_from_df(cols='age', label_cls=FloatList)
                  .databunch())
data.train_dl.batch_size = len(data.train_dl.dl.dataset)
data.train_dl = data.train_dl.new(shuffle=False)
train_x, train_y = next(iter(data.train_dl))

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
embedding_sizes = data.get_emb_szs({})
model = TabularGPModel(nb_continuous_inputs=len(data.cont_names), embedding_sizes=embedding_sizes, 
                       output_size=data.c, train_x=train_x, train_y=train_y, likelihood=likelihood)

# using the GPU
train_x = [train_x[0].cuda(), train_x[1].cuda()]
train_y = train_y.cuda()
model = model.cuda()
likelihood = likelihood.cuda()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 10
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(*train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
