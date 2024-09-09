import numpy as np
from scipy.stats import f, beta, chi2
import copy
import pandas as pd
import logging
import altair as alt
from .exceptions import NotDataFrameError, NComponentsError, NotAListError, ModelNotFittedError
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
   
    if observation.shape[1] != len(pca_model._variables):
        raise ValueError(f'Number of features in data must be {len(pca_model._variables)}')
    
    if observation.shape[0] != 1:
        raise ValueError(f'Number of observations in data must be 1')

    contributions_df, SPE = pca_model.spe_contribution(observation)

    # Altair plot for the residuals
    return alt.Chart(contributions_df).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y('contribution', title='Contribution'),
        tooltip=['variable', 'contribution']
    ).properties(
        title=f'Contribution to the SPE for the observation: {str(observation.index.values[0])} - SPE: {SPE[0]:.2f}'
    ).interactive(), contributions_df

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
    
    if observation.shape[1] != len(pca_model._variables):
        raise ValueError(f'Number of features in data must be {len(pca_model._variables)}')
    
    if observation.shape[0] != 1:
        raise ValueError(f'Number of observations in data must be 1')
    
    contributions_df, hotelling = pca_model.t2_contribution(observation)

    # Altair plot for the residuals
    return alt.Chart(contributions_df).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y('contribution', title='Contribution'),
        tooltip=['variable', 'contribution']
    ).properties(
        title=f'Contribution to the Hotelling\'s T2 of observation {str(observation.index.values[0])} - \n T2: {hotelling[0]:.2f}'
    ).interactive(), contributions_df
