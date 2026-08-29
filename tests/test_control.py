import numpy as np
import pandas as pd
from pca_tools.model import PCA

def test_control_limits(sample_data):
    # Setup
    alpha = 0.95
    pca_model = PCA(n_comps=2)
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
    """Phase II should distinguish latent-space and residual-space deviations."""
    model = PCA(n_comps=2, alpha=0.99).fit(sample_data)
    component_columns = model._scores.columns

    # A large score lies in the fitted latent subspace: it should trigger T²,
    # while retaining a near-zero reconstruction residual.
    latent_scores = np.zeros((1, model._ncomps))
    latent_scores[0, 0] = 50
    latent_anomaly = model.inverse_transform(pd.DataFrame(latent_scores, columns=component_columns))

    # Construct a vector orthogonal to the loading matrix. Adding it in the
    # standardized feature space produces a pure residual (SPE) deviation.
    _, _, right_singular_vectors = np.linalg.svd(model._loadings.values.T, full_matrices=True)
    residual_direction = right_singular_vectors[model._ncomps]
    in_control = model.inverse_transform(pd.DataFrame(np.zeros((1, model._ncomps)), columns=component_columns))
    standardized = model.preprocess(in_control)
    residual_space = standardized + 50 * residual_direction
    residual_anomaly = pd.DataFrame(
        model._scaler.inverse_transform(residual_space),
        columns=sample_data.columns,
    )

    phase_ii = pd.concat([in_control, latent_anomaly, residual_anomaly], ignore_index=True)
    hotelling_t2, spe, _, _ = model.project(phase_ii)

    assert hotelling_t2[0] < model.control_limits_["T2_phase2"]
    assert spe[0] < model.control_limits_["SPE"]
    assert hotelling_t2[1] > model.control_limits_["T2_phase2"]
    assert spe[1] < model.control_limits_["SPE"]
    assert hotelling_t2[2] < model.control_limits_["T2_phase2"]
    assert spe[2] > model.control_limits_["SPE"]
