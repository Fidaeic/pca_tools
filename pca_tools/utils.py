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

def pca_tsr_imputation_sk(data, n_components=None, max_iter=100, tol=1e-6):
    """
    Perform PCA-based imputation using Total Statistical Reconstruction (TSR).
    
    Parameters:
    - data: pandas DataFrame with missing values (NaNs)
    - n_components: Number of principal components to retain (default: None, keeps all components)
    - max_iter: Maximum number of iterations (default: 100)
    - tol: Tolerance for convergence (default: 1e-6)
    
    Returns:
    - Imputed DataFrame
    """
    df = data.copy()
    missing_mask = df.isna()
    
    df_filled = df.fillna(df.mean())  # Initial guess using column means
    
    scaler = StandardScaler()
    df_filled_scaled = scaler.fit_transform(df_filled)
    
    for iteration in range(max_iter):
        # Apply PCA
        pca = skPCA(n_components=n_components)
        pca.fit(df_filled_scaled)
        df_projected_scaled = pca.inverse_transform(pca.transform(df_filled_scaled))
        
        # Destandardize the data
        df_projected = pd.DataFrame(scaler.inverse_transform(df_projected_scaled), columns=df.columns, index=df.index)
        
        # Replace missing values with reconstructed values
        df_filled_new = df_filled.copy()
        df_filled_new[missing_mask] = df_projected[missing_mask]
        
        # Standardize the new filled data
        df_filled_new_scaled = scaler.transform(df_filled_new)
        
        # Check for convergence using scaled data
        if np.linalg.norm(df_filled_new_scaled - df_filled_scaled) < tol:
            break
        
        df_filled = df_filled_new
        df_filled_scaled = df_filled_new_scaled

    return pd.DataFrame(df_filled, columns=df.columns, index=df.index)