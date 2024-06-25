import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pca_tools.model import PCA  # Replace with the actual import
from pandas.testing import assert_frame_equal

def test_preprocess_numerical(preprocess_data):

    data = preprocess_data
    
    expected_result = pd.read_parquet('./tests/references/preprocessed_numerical.parquet')

    # Assuming YourClass contains the preprocess method
    obj = PCA(n_comps=1, numerical_features=['numerical'])

    # Fit the data to fit the scaler
    obj.fit(data)
    
    # Preprocess the data
    result = obj.preprocess(data)
    
    # Assertions
    assert_frame_equal(result, expected_result)

def test_preprocess_all(preprocess_data):

    data = preprocess_data
    
    expected_result = pd.read_parquet('./tests/references/preprocessed_all.parquet')

    # Assuming YourClass contains the preprocess method
    obj = PCA(n_comps=1)

    # Fit the data to fit the scaler
    obj.fit(data)
    
    # Preprocess the data
    result = obj.preprocess(data)
    
    # Assertions
    assert_frame_equal(result, expected_result)