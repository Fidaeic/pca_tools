#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:40:07 2021

@author: fidae
"""

import numpy as np
from numpy import linalg as LA
from scipy.stats import f, beta, chi2
import copy
from .model import PCA
import pandas as pd
import logging
import altair as alt

def optimize(X, n_comps, alpha, numerical_features, statistic='T2', threshold=3):
    """
    This function optimizes a dataset for PCA by iteratively removing out-of-control observations.

    Parameters
    ----------
    X : pd.DataFrame
        The input data to be optimized.
    n_comps : int
        The number of principal components to use in the PCA.
    alpha : float
        The significance level for the control limits. Must be between 0 and 1.
    numerical_features : list
        The list of numerical features to be considered in the PCA.
    statistic : str, optional
        The statistic to use for determining out-of-control observations. Must be either 'T2' for Hotelling's T^2 or 'SPE' for Squared Prediction Error. Default is 'T2'.
    threshold : float, optional
        The threshold for determining out-of-control observations. Observations with a statistic value greater than threshold times the control limit are considered out-of-control. Default is 3.

    Returns
    -------
    X_opt : pd.DataFrame
        The optimized input data with out-of-control observations removed.

    Raises
    ------
    ValueError
        If `statistic` is not 'T2' or 'SPE', `alpha` is not between 0 and 1, `threshold` is not positive, `n_comps` is not greater than 0, `numerical_features` is not a list, or `X` is not a pandas DataFrame.
    """

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Validate inputs
    if statistic not in ['T2', 'SPE']:
        raise ValueError("Statistic must be 'T2' or 'SPE'")
    
    if alpha<0 or alpha>1:
        raise ValueError("Alpha must be between 0 and 1")
    
    if threshold<0:
        raise ValueError("Threshold must be positive")
    
    if n_comps<1:
        raise ValueError("Number of components must be greater than 0")
    
    if not isinstance(numerical_features, list):
        raise ValueError("Numerical features must be a list")
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    # Initialize variables
    keep_indices = np.arange(X.shape[0])

    # Train PCA model
    pca = PCA(n_comps=n_comps)
    pca.train(X, numerical_features=numerical_features, alpha=alpha)

    # Determine control limit and statistic value based on statistic
    if statistic == 'T2':
        control_limit = pca._hotelling_limit_p1
        statistic_value = pca._hotelling
    else:
        control_limit = pca._spe_limit
        statistic_value = pca._spe

    # Calculate proportion of out-of-control observations
    out_of_control = np.sum(statistic_value > control_limit) / len(keep_indices)

    # Iteratively remove out-of-control observations
    while out_of_control > (1 - alpha):
        # Identify out-of-control observations
        drop_indices = np.where(statistic_value > threshold * control_limit)[0]

        # If no observations to drop, break the loop
        if len(drop_indices) == 0:
            break

        # Remove out-of-control observations
        keep_indices = np.delete(keep_indices, drop_indices)

        # Retrain PCA model
        pca.train(X.iloc[keep_indices], numerical_features=numerical_features, alpha=alpha)

        # Recalculate control limit and statistic value
        if statistic == 'T2':
            control_limit = pca._hotelling_limit_p1
            statistic_value = pca._hotelling
        else:
            control_limit = pca._spe_limit
            statistic_value = pca._spe

        # Calculate proportion of out-of-control observations
        out_of_control = np.sum(statistic_value > control_limit) / len(keep_indices)

        # Log progress
        logging.info(f'Processing statistics: {statistic}')
        logging.info(f"{len(drop_indices)} removed observations")
        logging.info(f"Proportion of out of control observations: {out_of_control}")
        logging.info(f"Control limit: {control_limit}")

    # Return optimized data
    return X.iloc[keep_indices]

def spe_contribution_plot(pca_model, observation):

    '''
    Generates a bar plot of the contribution of each variable to the SPE statistic of the selected observation

    Parameters
    ----------
    obs : int
        The number of the observation.

    Returns
    -------
    None
    '''
    # if obs < 0 or obs >= self._nobs:
    #     raise ValueError("The observation number must be between 0 and the number of observations")

    # Calculate the residuals based on the model
    if not isinstance(observation, pd.DataFrame):
            raise ValueError(f'Data must be of type pandas DataFrame, not {type(observation)}')
    
    if observation.shape[1] != len(pca_model._variables):
        raise ValueError(f'Number of features in data must be {len(pca_model._variables)}')
    
    if observation.shape[0] != 1:
        raise ValueError(f'Number of observations in data must be 1')
    
    pca_copy = copy.deepcopy(pca_model)

    _, SPE, residuals, _ = pca_copy.predict(observation)

    residuals = pd.DataFrame({'variable': pca_copy._variables, 'contribution': residuals[0]**2})

    # Altair plot for the residuals
    return alt.Chart(residuals).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y('contribution', title='Contribution'),
        tooltip=['variable', 'contribution']
    ).properties(
        title=f'Contribution to the SPE for the observation: {str(observation.index.values[0])} - SPE: {SPE[0]:.2f}'
    ).interactive()

def hotelling_t2_contribution_plot(pca_model, observation):
    '''
    Generates a bar plot of the contribution of each variable to the Hotelling's T2 statistic of the selected observation

    Parameters
    ----------
    obs : int
        The number of the observation.

    Returns
    -------
    None
    '''
    if not isinstance(observation, pd.DataFrame):
            raise ValueError(f'Data must be of type pandas DataFrame, not {type(observation)}')
    
    if observation.shape[1] != len(pca_model._variables):
        raise ValueError(f'Number of features in data must be {len(pca_model._variables)}')
    
    if observation.shape[0] != 1:
        raise ValueError(f'Number of observations in data must be 1')
    
    pca_copy = copy.deepcopy(pca_model)

    hotelling, _, _, _ = pca_copy.predict(observation)

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


def is_anomaly(pca_model, df_obs):

    pca_copy = copy.deepcopy(pca_model)

    hotelling, SPE, _, _ = pca_copy.predict(df_obs)

    # Hotelling's T2 control limit. Phase II
    dfn = pca_copy._ncomps
    dfd = pca_copy._nobs - pca_copy._ncomps
    const = (pca_copy._ncomps * (pca_copy._nobs**2 -1)) / (pca_copy._nobs * (pca_copy._nobs - pca_copy._ncomps))

    prob_hotelling = 1-f.cdf(hotelling/const, dfn, dfd)

    # SPE control limit. Phase II
    b, nu = np.mean(pca_copy._spe), np.var(pca_copy._spe)
        
    df = (2*b**2)/nu
    const = nu/(2*b)

    prob_spe = 1-chi2.cdf(SPE/const, df)

    if SPE <= pca_copy._spe_limit and hotelling <= pca_copy._hotelling_limit_p2:
        print("No anomaly detected.")

    if SPE > pca_copy._spe_limit:
        print("Moderate outlier detected.")
        print("SPE: ", SPE)
        print("SPE limit: ", pca_copy._spe_limit)
        print("Probability of observing this event: ", prob_spe)
    
    if hotelling > pca_copy._hotelling_limit_p2:
        print("Severe outlier detected.")
        print("Hotelling T2: ", hotelling)
        print("Hotelling T2 limit: ", pca_copy._hotelling_limit_p2)
        print("Probability of observing this event: ", prob_hotelling)