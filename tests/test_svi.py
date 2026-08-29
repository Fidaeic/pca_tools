import numpy as np

from pca_tools import PCA


def test_svi_metrics_are_bounded_monotonic_and_complete(sample_data):
    model = PCA(n_comps=sample_data.shape[1]).fit(sample_data)
    alpha = model.svi_["self_explanatory_power"]
    r2 = model.svi_["R2"]

    assert ((alpha >= 0) & (alpha <= 1)).all().all()
    assert ((r2 >= 0) & (r2 <= 1)).all().all()
    assert (np.diff(alpha.to_numpy(), axis=1) >= -1e-12).all()
    assert (np.diff(r2.to_numpy(), axis=1) >= -1e-12).all()
    np.testing.assert_allclose(alpha.iloc[:, -1], 1.0, atol=1e-10)
    np.testing.assert_allclose(r2.iloc[:, -1], 1.0, atol=1e-10)


def test_svi_plot_uses_interpretable_metric_labels(sample_data):
    model = PCA(n_comps=2).fit(sample_data)

    specification = model.svi_plot(sample_data.columns[0]).to_dict()

    assert "Self-explanatory power" in str(specification)
    assert "Variance explained" in str(specification)
