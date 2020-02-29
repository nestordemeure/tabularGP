from fastai.tabular import *
from tabularGP import tabularGP_learner

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
