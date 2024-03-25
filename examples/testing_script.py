#%%
import os
import pandas as pd
import sys
sys.path.append("pca_tools/")
from model import PCA
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA as skPCA

# %%
data = pd.read_csv('data/water_potability.csv')

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

data_transform = pd.DataFrame(pipeline.fit_transform(data), columns=data.columns)
# %%
pca = PCA(verbose=True, demean=False, descale=False, tolerance=1e-4)
pca.fit(data_transform, 4)
#%%
sk_pca = skPCA(n_components=4)
sk_pca.fit(data_transform)
#%%
pca.hotelling_t2_plot(.99)
# %%
pca.spe_contribution_plot(287)
# %%
(pca._residuals_fit[287]**2).sum()
# %%
contrib = ((pca._loadings.values*pca._training_data[1554])/pca._eigenvals[:, None])**2
# %%
import seaborn as sns
sns.barplot(np.sum(contrib, axis=1))
# %%
pca.hotelling_t2_contribution_plot(1554)
# %%
pca.difference_plot(1554)
# %%
pca.biplot(1, 2)
# %%
pca.loadings_barplot(1)
# %%
