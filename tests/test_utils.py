import numpy as np
import pandas as pd
import pytest

from pca_tools import column_wise_k_fold_pca_cv, pca_imputation


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


def test_column_wise_cv_caps_default_components_to_the_smallest_training_fold():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(30, 2))
    data = pd.DataFrame(latent @ rng.normal(size=(2, 5)))

    optimal, press_scores = column_wise_k_fold_pca_cv(data, n_splits=3)

    assert 1 <= optimal <= 3
    assert 1 <= len(press_scores) <= 3
    assert np.isfinite(press_scores).all()


def test_column_wise_cv_rejects_infeasible_component_count():
    data = pd.DataFrame(np.arange(30, dtype=float).reshape(6, 5))

    with pytest.raises(ValueError, match="max_components"):
        column_wise_k_fold_pca_cv(data, n_splits=3, max_components=4)
