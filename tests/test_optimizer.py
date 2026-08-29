import pytest

from pca_tools import PCA, PCAOptimizer


def test_lightweight_fit_keeps_monitoring_results(sample_data):
    model = PCA(n_comps=2).fit(sample_data, compute_diagnostics=False)

    assert set(model.phase1_statistics_) == {"T2", "SPE"}
    assert set(model.control_limits_) == {"T2_phase1", "T2_phase2", "SPE", "DModX"}
    assert not hasattr(model, "_alpha_A")


def test_optimizer_records_an_auditable_result(sample_data):
    optimizer = PCAOptimizer(
        n_comps=2,
        alpha=0.95,
        statistic="both",
        max_outlier_fraction=0.5,
    )

    curated = optimizer.optimize(sample_data)

    assert curated.equals(optimizer.result_.in_control_data)
    assert optimizer.result_.model.is_fitted()
    assert optimizer.result_.history
    assert optimizer.result_.chart_alpha == pytest.approx(0.975)


def test_optimizer_rejects_spe_without_a_residual_subspace(sample_data):
    optimizer = PCAOptimizer(
        n_comps=sample_data.shape[1],
        alpha=0.95,
        statistic="SPE",
    )

    with pytest.raises(ValueError, match="SPE monitoring requires"):
        optimizer.optimize(sample_data)
