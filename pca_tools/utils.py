import numpy as np
from scipy.stats import f, beta, chi2
import copy
from .model import PCA
import pandas as pd
import logging
import altair as alt
from .exceptions import NotDataFrameError, NComponentsError, NotAListError, ModelNotFittedError

def optimize(X, n_comps, alpha, numerical_features, statistic='T2', threshold=3, drop_percentage=0.2):
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
    
    if not isinstance(X, pd.DataFrame):
            raise NotDataFrameError(type(X).__name__)
    
    if n_comps <= 0 or n_comps>X.shape[1]:
            raise NComponentsError(X.shape[1])
    
    if not isinstance(numerical_features, list):
        raise NotAListError()

    # Initialize variables
    keep_indices = np.arange(X.shape[0])

    # Train PCA model
    pca = PCA(n_comps=n_comps, numerical_features=numerical_features, alpha=alpha)
    pca.fit(X)

    # Determine control limit and statistic value based on statistic
    if statistic == 'T2':
        control_limit = pca._hotelling_limit_p1
        statistic_value = np.array(pca._hotelling)
    else:
        control_limit = pca._spe_limit
        statistic_value = np.array(pca._spe)

    # Calculate proportion of out-of-control observations
    out_of_control = np.sum(statistic_value > control_limit) / len(keep_indices)

    n_dropped = 1
    # Iteratively remove out-of-control observations
    while out_of_control > (1 - alpha)*threshold and n_dropped>0:
        # Identify out-of-control observations
        out_of_control_indices = np.where(statistic_value > control_limit)[0]

        # If no observations to drop, break the loop
        if len(out_of_control_indices) == 0:
            break

        # Sort out-of-control observations by statistic value in descending order
        sorted_indices = out_of_control_indices[np.argsort(statistic_value[out_of_control_indices])[::-1]]

        # Select top 10% of out-of-control observations
        drop_indices = sorted_indices[:int(len(sorted_indices) * drop_percentage)]

        n_dropped = len(drop_indices)

        # Remove top 10% out-of-control observations
        keep_indices = np.delete(keep_indices, drop_indices)

        # Retrain PCA model
        pca.fit(X.iloc[keep_indices],)

        # Recalculate control limit and statistic value
        if statistic == 'T2':
            control_limit = pca._hotelling_limit_p1
            statistic_value = np.array(pca._hotelling)
        else:
            control_limit = pca._spe_limit
            statistic_value = np.array(pca._spe)

        # Calculate proportion of out-of-control observations
        out_of_control = np.sum(statistic_value > control_limit) / len(keep_indices)

        # Log progress
        logging.info(f'Processing statistics: {statistic}')
        logging.info(f"{n_dropped} removed observations")
        logging.info(f"Proportion of out of control observations: {out_of_control}")
        logging.info(f"Control limit: {control_limit}")

    # Return optimized data
    return X.iloc[keep_indices]

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