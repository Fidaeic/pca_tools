import pandas as pd
import numpy as np
from pca_tools.model import PCA 
from pandas.testing import assert_frame_equal

def test_scores(sample_data):

    expected_scores = pd.read_parquet('./tests/references/scores.parquet')
    
    pca = PCA(n_comps=3, standardize=True, tolerance=0.00001)
    pca.fit(sample_data)

    assert_frame_equal(np.abs(expected_scores), np.abs(pca._scores), atol=1e-3)
    assert_frame_equal(np.abs(expected_scores), np.abs(pca.transform(sample_data)), atol=1e-3)
    assert_frame_equal(np.abs(expected_scores), np.abs(pca.fit_transform(sample_data)), atol=1e-3)

def test_loadings(sample_data):
    
    expected_loadings = pd.read_parquet('./tests/references/loadings.parquet')
    
    pca = PCA(n_comps=3, standardize=True, tolerance=0.00001)
    pca.fit(sample_data)

    assert_frame_equal(np.abs(expected_loadings), np.abs(pca._loadings), atol=1e-3)

def test_inverse_transform(sample_data):

    pca = PCA(n_comps=sample_data.shape[1], standardize=True, tolerance=0.00001)
    pca.fit(sample_data)

    assert_frame_equal(sample_data, pca.inverse_transform(pca.transform(sample_data)), atol=1e-3)

def test_inverse_transform_no_standardization(sample_data):

    pca = PCA(n_comps=sample_data.shape[1], standardize=False, tolerance=0.00001)
    pca.fit(sample_data)

    assert_frame_equal(sample_data, pca.inverse_transform(pca.transform(sample_data)), atol=1e-3)

def test_project_method(sample_data):
    pca = PCA(n_comps=2, standardize=False, tolerance=0.00001)
    pca.fit(sample_data)

    hotelling_t2, spe_p2, residuals, predicted_scores = pca.project(sample_data)

    # Debugging: Print the types of the outputs for verification
    print(f"Types - Hotelling's T2: {type(hotelling_t2)}, SPE: {type(spe_p2)}, Residuals: {type(residuals)}, Predicted Scores: {type(predicted_scores)}")

    
    # Validate the types of the outputs
    assert isinstance(hotelling_t2, list), "Hotelling's T2 should be a list"
    assert isinstance(spe_p2, list), "SPE should be a list"
    assert isinstance(residuals, np.ndarray), "Residuals should be an np.ndarray"
    assert isinstance(predicted_scores, pd.DataFrame), "Predicted scores should be a pd.DataFrame"