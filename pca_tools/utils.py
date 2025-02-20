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

def column_wise_k_fold_pca_cv(data: pd.DataFrame, 
                              max_components: int | None = None, 
                              n_splits: int = 5,
                              improvement_tol: float = 0.01) -> tuple[int, list[float]]:
    """
    Perform column-wise k-fold (CKF) cross-validation to select the optimal number of principal components,
    with an early stopping mechanism based on the relative improvement of PRESS between consecutive runs.
    
    In this approach, the columns of the data are partitioned into n_splits groups. For each group,
    the PCA model is trained on the remaining columns, and then the left-out columns are predicted via
    linear regression on the PCA scores. The reconstruction error for the left-out columns is computed
    as the Prediction Error Sum of Squares (PRESS). Instead of testing all possible components, the algorithm
    stops early if the relative improvement between consecutive candidate runs falls below improvement_tol.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data for PCA cross-validation. Rows with any missing values are dropped.
    max_components : int, optional
        Maximum number of principal components to test. If None, uses the number of columns.
    n_splits : int, optional
        Number of column-wise folds (default is 5).
    improvement_tol : float, optional
        The minimum relative improvement in PRESS (between consecutive candidates) required to continue;
        if the improvement is lower, the process stops (default is 0.01, i.e. 1% improvement).
    
    Returns
    -------
    tuple[int, list[float]]
        A tuple where the first element is the optimal number of principal components and the second element
        is a list of PRESS scores (one for each candidate number of components evaluated).
    
    Raises
    ------
    ValueError
        If the input data does not have enough columns.
    
    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> data = pd.DataFrame(np.random.rand(100, 10))
    >>> optimal, scores = column_wise_k_fold_pca_cv(data, max_components=5, n_splits=3, improvement_tol=0.01)
    >>> print(optimal)
    3
    """
    # Drop rows with missing values to ensure PCA can be applied.
    df = data.copy().dropna(axis=0)
    if df.shape[1] < n_splits:
        raise ValueError("Number of splits exceeds number of columns in data.")
    
    # Standardize the data.
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    num_cols = df.shape[1]
    # Split column indices into n_splits groups.
    col_indices = np.array_split(np.arange(num_cols), n_splits)

    max_components = max_components or num_cols
    press_scores = []

    optimal_components = max_components  # Default to max_components if early stopping never triggers.
    
    for n_components in range(1, max_components + 1):
        total_press = 0

        for col_group in col_indices:
            # Training data: remove left-out columns.
            df_train = np.delete(df_scaled, col_group, axis=1)
            # Test data: left-out columns.
            df_test = df_scaled[:, col_group]

            # Fit PCA on training data.
            pca = PCA_sk(n_components=n_components)
            T = pca.fit_transform(df_train)  # Scores: shape (n_samples, n_components)

            # Predict test columns via linear regression by solving T * B = df_test.
            B, _, _, _ = np.linalg.lstsq(T, df_test, rcond=None)
            X_test_hat = T @ B

            # Compute PRESS for the left-out columns.
            error = df_test - X_test_hat
            press = np.sum(np.square(error))
            total_press += press

        press_scores.append(total_press)

        # If not the first candidate, check relative improvement.
        if n_components > 1:
            improvement = press_scores[-2] - press_scores[-1]
            rel_improvement = improvement / press_scores[-2]
            if rel_improvement < improvement_tol:
                optimal_components = n_components - 1
                # Optional: break early here if further improvement is negligible.
                break
    else:
        # If loop completes normally, choose the candidate with minimum PRESS.
        optimal_components = int(np.argmin(press_scores) + 1)

    return optimal_components, press_scores

def compute_R2_matrix(P, X):
    """
    Compute the cumulative R² values for each variable as more principal components are included.

    This function calculates a cumulative R² metric for each variable by iteratively considering an increasing number of principal components (columns of the loadings matrix P). For each component index a (ranging
    from 0 to A-1), the followingd steps are performed:
    
    1. Compute a temporary squared loadings matrix Q_A using the first (a+1) columns of P: 
         Q_A = P[:, :a+1] @ P[:, :a+1].T  
       This results in an (M x M) matrix, where M is the number of variables.
       
    2. Extract the diagonal of Q_A (denoted as alpha_A), which represents the direct self-explanatory 
       contribution of each variable when using the first (a+1) components.
       
    3. For each variable m, compute a numerator that aggregates the effect of the variable’s own contribution 
       (term1) and its cross-interactive contributions with all other variables (term2). Specifically:
         - term1 is computed as X[:, m] * alpha_A[m],
         - term2 is the sum over all other variables v (v ≠ m) of the product X[:, v] * Q_A[v, m],
         - The numerator is then the sum of squares of (term1 + term2) across all samples.
       
    4. Compute the denominator as the sum of squares of X[:, m] over all samples.
       
    5. The cumulative R² value for variable m with (a+1) components is given by:
         R2[m, a] = numerator / denominator   (if denominator is non-zero, otherwise 0).

    Parameters
    ----------
    P : numpy.ndarray
        Loadings matrix from PCA of shape (M, A), where M is the number of variables and A is the total number 
        of principal components.
    X : numpy.ndarray
        Data matrix of shape (N, M), assumed to be centered and normalized, where N is the number of samples.

    Returns
    -------
    numpy.ndarray
        A matrix of cumulative R² values with shape (M, A), where each row corresponds to a variable and each 
        column corresponds to the cumulative R² computed using the first (a+1) principal components.
    """
    _, M = X.shape
    A = P.shape[1]  # Total number of principal components
    R2 = np.zeros((M, A))  # Initialize the result matrix

    # Iterate over the number of principal component subsets (cumulative)
    for a in range(A):
        # Compute Q_A using the first (a+1) components of P, resulting in an (M x M) matrix
        Q_A = P[:, :a+1] @ P[:, :a+1].T
        
        # The diagonal of Q_A gives the self-explanatory power for the current set of components
        alpha_A = np.diag(Q_A)
        
        # For each variable, compute the cumulative R² metric
        for m in range(M):
            # term1: Contribution of variable m itself
            term1 = X[:, m] * alpha_A[m]
            # term2: Contribution from the interactions with other variables
            term2 = np.sum([X[:, v] * Q_A[v, m] for v in range(M) if v != m], axis=0)
            # Numerator: Sum of squared aggregated contributions over all samples
            numerator = np.sum((term1 + term2) ** 2)
            # Denominator: Sum of squares of the values for variable m
            denominator = np.sum(X[:, m] ** 2)
            
            # Calculate cumulative R² for variable m for the current number of components
            R2[m, a] = numerator / denominator if denominator != 0 else 0

    return R2