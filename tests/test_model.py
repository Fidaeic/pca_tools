import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pca_tools.model import PCA  # Replace with the actual import
from pandas.testing import assert_frame_equal

def test_train(sample_data):
    pca = PCA(n_comps=2, standardize=True, tolerance=0.001, verbose=False)
    pca.fit(sample_data)

    # Check if the model has been fitted
    assert hasattr(pca, '_eigenvals'), "Model should compute eigenvalues"
    assert len(pca._eigenvals) == 2, "There should be 2 eigenvalues for 2 components"
    
    assert isinstance(pca._loadings, pd.DataFrame), "Loadings should be a pandas DataFrame"
    assert pca._loadings.shape == (5, 2), "Loadings shape should match the number of components and features"
    
    assert isinstance(pca._scores, pd.DataFrame), "Scores should be a pandas DataFrame"
    assert pca._scores.shape[0] == 100, "Scores should have the same number of rows as the input data"
    
    assert len(pca._explained_variance) == 2, "Explained variance should have an entry for each component"


def test_explained_variance(sample_data):
    
    pca = PCA(n_comps=sample_data.shape[1], standardize=True, tolerance=0.00001)
    pca.fit(sample_data)

    assert pca._rsquared_acc[-1] == 1
