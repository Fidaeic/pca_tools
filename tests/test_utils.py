import numpy as np
import pandas as pd
import pytest

from pca_tools import pca_imputation


def test_pca_imputation_preserves_observations_and_imputes_missing_values():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, np.nan, 10.0],
        }
    )

    result = pca_imputation(data, n_components=1)

    assert result.index.equals(data.index)
    assert result.columns.equals(data.columns)
    assert not result.isna().any().any()
    assert result.where(data.notna()).equals(data.where(data.notna()))
    # This also catches returning the initial mean-filled estimate on convergence.
    assert not np.isclose(result.loc[2, "x"], data["x"].mean())


def test_pca_imputation_rejects_a_column_without_observations():
    data = pd.DataFrame({"observed": [1.0, 2.0], "missing": [np.nan, np.nan]})

    with pytest.raises(ValueError, match="no observed values"):
        pca_imputation(data, n_components=1)
