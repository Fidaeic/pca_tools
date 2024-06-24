import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

@pytest.fixture(scope="module")
def preprocess_data():
    # Setup
    data = pd.DataFrame({
        'numerical': [1, 2, 3, 4, 5],
        'non_numerical': [0, 1, 1, 0, 1]
    })

    return data

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)  # For reproducibility
    data = np.random.rand(100, 5)  # 100 samples, 5 features
    columns = [f'feature_{i}' for i in range(1, 6)]
    return pd.DataFrame(data, columns=columns)