import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler

@pytest.fixture(scope="module")
def preprocess_data():
    # Setup
    data = pd.DataFrame({
        'numerical': [1, 2, 3, 4, 5],
        'non_numerical': [0, 1, 1, 0, 1]
    })

    return data