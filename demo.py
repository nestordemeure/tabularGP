from fastai.tabular import *
from tabularGP import tabularGP_learner

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv').sample(1000)
procs = [FillMissing, Normalize, Categorify]

#------------------------------------------------------------------------------
# Classification

# features
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['education-num', 'fnlwgt', 'age']

dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_rand_pct()
                  .label_from_df(cols='salary')
                  .databunch(bs=63))

# gp model
glearn = tabularGP_learner(dls, nb_training_points=50, metrics=accuracy)
glearn.fit_one_cycle(10, max_lr=1e-2)

# classical model
#learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
#learn.fit_one_cycle(10, max_lr=1e-2)

#------------------------------------------------------------------------------
# Regression

# features
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'salary']
cont_names = ['education-num', 'fnlwgt']

# load data
dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_rand_pct()
                  .label_from_df(cols='age', label_cls=FloatList)
                  .databunch())

# gp model
glearn = tabularGP_learner(dls, nb_training_points=50, metrics=[rmse, mae])
glearn.fit_one_cycle(10, max_lr=1e-1)

# classical model
#learn = tabular_learner(dls, layers=[200,100], metrics=[rmse, mae])
#learn.fit(10, 1e-2)

#------------------------------------------------------------------------------
# Multiple regression

# features
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'salary']
cont_names = ['education-num']

# load data
dls = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                  .split_by_rand_pct()
                  .label_from_df(cols=['age','fnlwgt'], label_cls=FloatList)
                  .databunch())

# gp model
glearn = tabularGP_learner(dls, nb_training_points=50, metrics=[rmse, mae])
glearn.fit_one_cycle(10, max_lr=1e-1)

# classical model
#learn = tabular_learner(dls, layers=[200,100], metrics=[rmse, mae])
#learn.fit_one_cycle(10, max_lr=1e-2)
