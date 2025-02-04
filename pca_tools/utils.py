import numpy as np
from scipy.stats import f, beta, chi2
import copy
import pandas as pd
import logging
import altair as alt
from numpy import linalg as LA
import pdb
from numba import jit
import functools
import pandas as pd
from .exceptions import NotDataFrameError, ModelNotFittedError
from sklearn.decomposition import PCA as PCA_sk
from sklearn.preprocessing import StandardScaler

def pca_imputation(data: pd.DataFrame, 
                   n_components: int | None = None, 
                   max_iter: int = 100, 
                   tol: float = 1e-6) -> pd.DataFrame:
    """
    Perform PCA-based imputation using Total Statistical Reconstruction (TSR).

    This function imputes missing values in the input DataFrame by iteratively reconstructing the data 
    via PCA. The missing values are first replaced by the column means. Then the data is standardized, 
    and PCA is applied. The reconstruction from PCA (inverse transformed back into the original space) 
    is used to update the missing entries. The process is iterated until the change between iterations 
    is less than the specified tolerance.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing missing values (NaNs) that need to be imputed.
    n_components : int, optional
        Number of principal components to retain. If None, keep all components. Default is None.
    max_iter : int, optional
        Maximum number of iterations for the imputation process. Default is 100.
    tol : float, optional
        Tolerance for convergence based on the change in standardized data. Default is 1e-6.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the missing values imputed.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [4.0, 5.0, np.nan]})
    >>> imputed = pca_imputation(data, n_components=1)
    >>> print(imputed)
    """
    # Make a copy of the original data and identify missing locations.
    df = data.copy()
    missing_mask = df.isna()

    # Initialize missing entries with column means.
    df_filled = df.fillna(df.mean())

    # Create and fit scaler on the initial guess.
    scaler = StandardScaler()
    df_filled_scaled = scaler.fit_transform(df_filled)

    for iteration in range(max_iter):
        # Apply PCA on scaled data.
        pca = PCA_sk(n_components=n_components)
        scores = pca.fit_transform(df_filled_scaled)
        reconstruction_scaled = pca.inverse_transform(scores)

        # Revert scaling back to the original data space.
        reconstruction = pd.DataFrame(
            scaler.inverse_transform(reconstruction_scaled), 
            columns=df.columns, index=df.index
        )

        # Only update missing values.
        df_filled_new = df_filled.copy()
        df_filled_new[missing_mask] = reconstruction[missing_mask]

        # Standardize the updated filled data.
        df_filled_new_scaled = scaler.transform(df_filled_new)

        # Check convergence based on the norm difference in standardized data.
        diff_norm = np.linalg.norm(df_filled_new_scaled - df_filled_scaled)
        if diff_norm < tol:
            break

        df_filled = df_filled_new
        df_filled_scaled = df_filled_new_scaled

    return pd.DataFrame(df_filled, columns=df.columns, index=df.index)