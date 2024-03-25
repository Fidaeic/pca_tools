#%%
import os
import pandas as pd
import sys
sys.path.append("src/")
from model import PCA

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# %%
data = pd.read_csv('data/water_potability.csv')

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

data_transform = pd.DataFrame(pipeline.fit_transform(data), columns=data.columns)
# %%
pca = PCA(ncomps=3, verbose=True, demean=False, descale=False, tolerance=1e-4)
pca.fit(data_transform)
#%%
pca.transform(data_transform)
#%%
pca.biplot(1, 2)
#%%
pca.loadings_barplot(2)
# %%

pca._scores
# %%
pca._loadings.T
# %%
# Barplot showing the difference between a specific observation and the mean of the sample
sample_mean = data.mean()
sample_std = data.std()
standardized_observation = (data.loc[1554] - sample_mean) / sample_std


standardized_observation.plot(kind='bar')
# data.mean().plot(kind='bar', color='orange')

# %%
sample_mean
# %%
data.loc[1554]