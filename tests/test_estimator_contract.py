import numpy as np
import pandas as pd
import pytest

from pca_tools import PCA


def test_fit_transform_is_a_complete_fit(sample_data):
    model = PCA(n_comps=2)

    scores = model.fit_transform(sample_data)

    assert model.is_fitted()
    assert scores.equals(model.transform(sample_data))
    assert hasattr(model, "_spe_limit")


def test_control_limits_respect_the_requested_alpha(sample_data):
    model = PCA(n_comps=2).fit(sample_data)

    limits_95 = model.control_limits(0.95)
    limits_99 = model.control_limits(0.99)

    assert limits_95 != limits_99


def test_set_params_reconfigures_the_estimator(sample_data):
    model = PCA(n_comps=2)
    model.set_params(n_comps=3, alpha=0.95)
    model.fit(sample_data)

    assert model.get_params()["n_comps"] == 3
    assert model._scores.shape[1] == 3
    assert model._alpha == 0.95


def test_transform_rejects_reordered_features(sample_data):
    model = PCA(n_comps=2).fit(sample_data)

    with pytest.raises(ValueError, match="feature names and order"):
        model.transform(sample_data[sample_data.columns[::-1]])


def test_default_model_preserves_a_residual_subspace(sample_data):
    model = PCA().fit(sample_data)

    assert model._ncomps == sample_data.shape[1] - 1
    assert np.isfinite(model._spe_limit)
    assert np.isfinite(model._dmodx_limit)


def test_full_rank_model_marks_residual_statistics_unavailable(sample_data):
    model = PCA(n_comps=sample_data.shape[1]).fit(sample_data)
    _, _, spe_limit, dmodx_limit = model.control_limits()

    assert np.isnan(spe_limit)
    assert np.isnan(dmodx_limit)
    with pytest.raises(ValueError, match="DModX is unavailable"):
        model.dmodx(sample_data)


def test_standardized_spe_is_invariant_to_feature_units(sample_data):
    scale = pd.Series([1.0, 10.0, 100.0, 0.1, 5.0], index=sample_data.columns)
    scaled_data = sample_data.mul(scale, axis="columns")

    baseline_spe, _ = PCA(n_comps=2, standardize=True).fit(sample_data).spe(sample_data)
    scaled_spe, _ = PCA(n_comps=2, standardize=True).fit(scaled_data).spe(scaled_data)

    np.testing.assert_allclose(baseline_spe, scaled_spe)
