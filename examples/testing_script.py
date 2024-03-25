#%%
import os
import pandas as pd
import sys
sys.path.append("src/")
from model import PCA
import numpy as np

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
pca = PCA(ncomps=4, verbose=True, demean=False, descale=False, tolerance=1e-4)
pca.fit(data_transform)
#%%
pca.spe_plot(.99)
# %%
pca.hotelling_t2_plot(.99)
# %%
