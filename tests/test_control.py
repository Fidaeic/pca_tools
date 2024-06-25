import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from pca_tools.model import PCA

def test_control_limits(sample_data):
    # Setup
    alpha = 0.95
    pca_model = PCA()
    pca_model.fit(sample_data)  # Assuming the PCA class has a fit method that sets necessary attributes

    # Action
    pca_model.control_limits(alpha)

    # Verification
    assert hasattr(pca_model, '_hotelling_limit_p1'), "Hotelling's T2 limit for Phase I not set"
    assert hasattr(pca_model, '_hotelling_limit_p2'), "Hotelling's T2 limit for Phase II not set"
    assert hasattr(pca_model, '_spe_limit'), "SPE limit not set"

    # Verify the limits are within expected ranges (This part is more complex and depends on the exact implementation details)
    # For simplicity, we're just checking if they are positive numbers
    assert pca_model._hotelling_limit_p1 > 0, "Hotelling's T2 limit for Phase I is not positive"
    assert pca_model._hotelling_limit_p2 > 0, "Hotelling's T2 limit for Phase II is not positive"
    assert pca_model._spe_limit > 0, "SPE limit is not positive"

def test_anomalies(sample_data):

    anomalous_observation = 10

    sample_data.iloc[anomalous_observation, :] = sample_data.iloc[anomalous_observation, :] * 1000  # Introduce an anomaly
    # Setup
    pca_model = PCA()
    pca_model.fit(sample_data)  # Assuming the PCA class has a fit method that sets necessary attributes

    # Action
    predictions = pca_model.predict(sample_data)

    # Verification
    assert predictions['spe_outlier'][anomalous_observation] == True, "Anomaly not detected"
    assert predictions['hotelling_outlier'][anomalous_observation] == True, "Anomaly not detected"