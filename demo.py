from fastai.tabular import *
from tabularGP import tabularGP_learner

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv').sample(1000)
procs = [FillMissing, Normalize, Categorify]

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
glearn = tabularGP_learner(dls, nb_training_points=50, metrics=[rmse, mae], fit_training_inputs=False, fit_training_outputs=False)
glearn.fit_one_cycle(5, max_lr=1e-1)

# active learning to improve the set of points used
glearn = tabularGP_learner(dls, nb_training_points=10, metrics=[rmse, mae], fit_training_inputs=False, fit_training_outputs=False)
glearn.fit_one_cycle(1, max_lr=1e-1)
glearn.active_learning2(nb_points=40)
glearn.fit_one_cycle(1, max_lr=1e-1)

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
glearnc = tabularGP_learner(dls, nb_training_points=50, metrics=accuracy)
glearnc.fit_one_cycle(10, max_lr=1e-3)

# feature importance
glearnc.plot_feature_importance()

# transfer learning on kernel
glearn = tabularGP_learner(dls, kernel=glearnc, nb_training_points=50, metrics=accuracy)
glearn.fit_one_cycle(10, max_lr=1e-3)

# transfer learning on prior
glearn = tabularGP_learner(dls, prior=glearnc, nb_training_points=50, metrics=accuracy)
glearn.fit_one_cycle(10, max_lr=1e-3)

# classical model
#learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
#learn.fit_one_cycle(10, max_lr=1e-2)
