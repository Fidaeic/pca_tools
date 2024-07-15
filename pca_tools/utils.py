import numpy as np
from scipy.stats import f, beta, chi2
import copy
import pandas as pd
import logging
import altair as alt
from .exceptions import NotDataFrameError, NComponentsError, NotAListError, ModelNotFittedError
import ray
from numpy import linalg as LA
import pdb
from numba import jit


def spe_contribution_plot(pca_model, observation:pd.DataFrame):

    '''
    Generates an interactive bar plot visualizing each variable's contribution to the Squared Prediction Error (SPE) for a specific observation.

    This function creates a bar plot that breaks down the contribution of each variable to the overall SPE of a given observation. It is useful for identifying which variables contribute most to the observation's deviation from the model's predictions. The function checks if the PCA model has been fitted and if the input observation is a valid single-row pandas DataFrame with the correct number of features.

    Parameters
    ----------
    pca_model : object
        The PCA model object that has been fitted to the data.
    observation : pd.DataFrame
        A single-row pandas DataFrame representing the observation to analyze.

    Raises
    ------
    ModelNotFittedError
        If the PCA model has not been fitted with data.
    NotDataFrameError
        If the input observation is not a pandas DataFrame.
    ValueError
        If the number of features in the observation does not match the model's expected number of features.
    ValueError
        If more than one observation is provided.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the interactive bar plot of variable contributions to the SPE.

    Notes
    -----
    - The function deep copies the PCA model to avoid altering its state.
    - The SPE and residuals are calculated using the model's `project` method.
    - The plot includes tooltips for each variable's contribution and displays the total SPE value in the title.
    '''

    if not hasattr(pca_model, '_scores'):
        raise ModelNotFittedError()

    if not isinstance(observation, pd.DataFrame):
        raise NotDataFrameError(type(observation).__name__)
    
    if observation.shape[1] != len(pca_model._variables):
        raise ValueError(f'Number of features in data must be {len(pca_model._variables)}')
    
    if observation.shape[0] != 1:
        raise ValueError(f'Number of observations in data must be 1')
    
    pca_copy = copy.deepcopy(pca_model)

    SPE, residuals = pca_copy.spe(observation)

    residuals = pd.DataFrame({'variable': pca_copy._variables, 'contribution': residuals[0]**2})

    # Altair plot for the residuals
    return alt.Chart(residuals).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y('contribution', title='Contribution'),
        tooltip=['variable', 'contribution']
    ).properties(
        title=f'Contribution to the SPE for the observation: {str(observation.index.values[0])} - SPE: {SPE[0]:.2f}'
    ).interactive()

def hotelling_t2_contribution_plot(pca_model, observation:pd.DataFrame):
    '''
    Generates an interactive bar plot visualizing each variable's contribution to the Hotelling's T2 statistic for a specific observation.

    This function creates a bar plot to illustrate how much each variable contributes to the Hotelling's T2 statistic of a given observation, aiding in the identification of variables that significantly influence the observation's deviation from the model's expectations. It validates the input observation to ensure it is a single-row pandas DataFrame with the correct number of features and checks if the PCA model has been fitted.

    Parameters
    ----------
    pca_model : object
        The PCA model object that has been fitted to the data.
    observation : pd.DataFrame
        A single-row pandas DataFrame representing the observation to analyze.

    Raises
    ------
    ModelNotFittedError
        If the PCA model has not been fitted with data.
    NotDataFrameError
        If the input observation is not a pandas DataFrame.
    ValueError
        If the number of features in the observation does not match the model's expected number of features.
    ValueError
        If more than one observation is provided.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the interactive bar plot of variable contributions to the Hotelling's T2 statistic.

    Notes
    -----
    - The function deep copies the PCA model to avoid altering its state.
    - Contributions are calculated based on the loadings and the observation's standardized values, normalized by the eigenvalues.
    - The plot includes tooltips for each variable's contribution and displays the total Hotelling's T2 value and the component with the maximum contribution in the title.
    '''
    if not hasattr(pca_model, '_scores'):
        raise ModelNotFittedError()

    if not isinstance(observation, pd.DataFrame):
        raise NotDataFrameError(type(observation).__name__)
    
    if observation.shape[1] != len(pca_model._variables):
        raise ValueError(f'Number of features in data must be {len(pca_model._variables)}')
    
    if observation.shape[0] != 1:
        raise ValueError(f'Number of observations in data must be 1')
    
    pca_copy = copy.deepcopy(pca_model)

    hotelling = pca_copy.hotelling_t2(observation)

    X_transform = observation.copy()

    if pca_copy._standardize:
        if pca_copy._numerical_features:
            X_transform[pca_copy._numerical_features] = pca_copy._scaler.transform(X_transform[pca_copy._numerical_features])
        else:
            X_transform = pd.DataFrame(pca_copy._scaler.transform(X_transform), columns=pca_copy._variables, index=pca_copy._index)
    
    contributions = (pca_copy._loadings.values*X_transform.values)
    normalized_contributions = (contributions/pca_copy._eigenvals[:, None])**2

    max_comp = np.argmax(np.sum(normalized_contributions, axis=1))

    contributions_df = pd.DataFrame({'variable': pca_copy._variables, 'contribution': contributions[max_comp]})

    # Altair plot for the residuals
    return alt.Chart(contributions_df).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y('contribution', title='Contribution'),
        tooltip=['variable', 'contribution']
    ).properties(
        title=f'Contribution to the Hotelling\'s T2 of observation {str(observation.index.values[0])} - \n T2: {hotelling[0]:.2f} - Comp: {max_comp}'
    ).interactive()

@ray.remote
def compute_component_ray(X_pca, X, tolerance, t_prev=None, cont=0, verbose=False):
    X_pca = X_pca.copy()
    column = np.argmax(np.var(X_pca, axis=0))
    t_prev = X_pca[:, column].reshape(-1, 1)
    cont = 0

    while True:
        # Compute p_t and t
        p_t = (t_prev.T @ X_pca) / (t_prev.T @ t_prev)
        p_t /= LA.norm(p_t)
        t = X_pca @ p_t.T

        # Check convergence
        conv = np.linalg.norm(t - t_prev)
        if verbose:
            print(f"Iteration {cont}, Convergence: {conv}")

        if conv <= tolerance:
            # Convergence achieved
            X_pca -= t @ p_t
            r2 = 1 - np.sum(X_pca**2) / np.sum(X**2)
            var_t = np.var(t)
            if verbose:
                print(f"Component converges after {cont} iterations")
            return X_pca, t.reshape(X_pca.shape[0]), p_t, var_t, r2
        else:
            # Prepare for next iteration
            t_prev = t
            cont += 1

def compute_component(X_pca, X, tolerance, t_prev=None, cont=0, verbose=False):
    X_pca = X_pca.copy()
    column = np.argmax(np.var(X_pca, axis=0))
    t_prev = X_pca[:, column].reshape(-1, 1)
    cont = 0

    while True:
        # Compute p_t and t
        p_t = (t_prev.T @ X_pca) / (t_prev.T @ t_prev)
        p_t /= LA.norm(p_t)
        t = X_pca @ p_t.T

        # Check convergence
        conv = np.linalg.norm(t - t_prev)
        if verbose:
            print(f"Iteration {cont}, Convergence: {conv}")

        if conv <= tolerance:
            # Convergence achieved
            X_pca -= t @ p_t
            r2 = 1 - np.sum(X_pca**2) / np.sum(X**2)
            var_t = np.var(t)
            if verbose:
                print(f"Component converges after {cont} iterations")
            return X_pca, t.reshape(X_pca.shape[0]), p_t, var_t, r2
        else:
            # Prepare for next iteration
            t_prev = t
            cont += 1
    

def compute_component_wrapper(use_ray, X_pca, X, tolerance, verbose=False, ray_workers=None):
    if use_ray:
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(num_cpus=ray_workers, ignore_reinit_error=True)
        # Call the function with Ray
        result_id = compute_component_ray.remote(X_pca, X, tolerance, verbose)
        return ray.get(result_id)
    else:
        # Call the function directly
        return compute_component(X_pca, X, tolerance, verbose)