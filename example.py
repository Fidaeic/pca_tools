#%%
import pandas as pd
from pca_tools import PCA
from sklearn.datasets import fetch_california_housing
from pca_tools.optimizer import PCAOptimizer
from sklearn.model_selection import train_test_split

import altair as alt
alt.data_transformers.enable("vegafusion")

ALPHA = .99
N_COMPS = 4
#%%
# Rows are in-control reference observations; columns are measured variables.
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
X_train, X_test = train_test_split(X)
# X.index.name = 'index'
model = PCA(n_comps=N_COMPS, alpha=ALPHA).fit(X_train)
print("Explained variance", model._rsquared_acc)
#%%
# Get the loadings barplot for a specified component
model.loadings_barplot(1)
# %%
model.biplot(1, 2)
#%%
# Check control limits and out-of-control observations
model.hotelling_t2_plot_p1()
#%%
model.spe_plot_p1()
#%%
# Optimize the dataset to keep the in-control observations given the specified parameters
opt = PCAOptimizer(n_comps=N_COMPS, alpha=ALPHA, numerical_features=data.feature_names)
# %%
X_opt = opt.optimize(X_train)
# %%
opt.result_.history
# %%
pca_opt = opt.result_.model
# %%
pca_opt.hotelling_t2_plot_p1()
#%%
pca_opt.spe_plot_p1()
# %%
opt.result_.removed_data
# %%
model.hotelling_t2_plot_p2(X_test)
# %%
pca_opt.hotelling_t2_plot_p2(X_test)
# %%
model.spe_plot_p2(X_test)
#%%
pca_opt.spe_plot_p2(X_test)
# %%
