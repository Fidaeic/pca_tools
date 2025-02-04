import altair as alt
import pandas as pd
import numpy as np

def score_plot(scores:pd.DataFrame, 
               comp1:int, 
               comp2:int, 
               explained_variance:np.ndarray,
               hue:pd.Series=None,
               index_name:str='index', 
               test_set:pd.DataFrame=None,):
        '''
        Generates a score plot of the selected components

        Parameters
        ----------
        comp1 : int
            The number of the first component.
        comp2 : int
            The number of the second component.

        Returns
        -------
        None
        '''       
        # Reset the index of the scores DataFrame
        scores = scores.reset_index()
        
        # Create horizontal and vertical lines at y=0 and x=0 respectively
        hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[12, 6]).encode(y='y').interactive()
        vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[12, 6]).encode(x='x').interactive()
        
        # Check if hue is provided
        if hue is not None:
            # Ensure hue is a pandas Series
            if not isinstance(hue, pd.Series):
                raise TypeError("Hue must be a pandas Series")
            
            # Add hue information to the scores DataFrame
            scores[hue.name] = hue
        
            # Create a scatter plot with color encoding based on hue
            scatter = alt.Chart(scores).mark_circle().encode(
                x=alt.X(f'PC_{comp1}',title=f'PC {comp1} - {explained_variance[comp1-1]*100:.2f} %'),
                y=alt.Y(f'PC_{comp2}',title=f'PC {comp2} - {explained_variance[comp2-1]*100:.2f} %'),
                tooltip=[f"PC_{comp1}", f"PC_{comp2}", hue.name],
                color=alt.Color(hue.name)
            ).interactive()
        
        else:
            # Create a scatter plot without color encoding
            scatter = alt.Chart(scores).mark_point().encode(
                x=alt.X(f'PC_{comp1}',title=f'PC {comp1} - {explained_variance[comp1-1]*100:.2f} %'),
                y=alt.Y(f'PC_{comp2}',title=f'PC {comp2} - {explained_variance[comp2-1]*100:.2f} %'),
                tooltip=[index_name, f"PC_{comp1}", f"PC_{comp2}"]
            ).interactive()
        
        # Check if test_set is provided
        if test_set is not None:
            # Reset the index of the test_set DataFrame
            scores_test = test_set.reset_index()
            
            # Create a scatter plot for the test set with black points
            scatter_test = alt.Chart(scores_test).mark_point(color='black', opacity=.1).encode(
                x=f"PC_{comp1}",
                y=f"PC_{comp2}",
                tooltip=[index_name, f"PC_{comp1}", f"PC_{comp2}"]
            ).interactive()
        
            # Return the combined plot of test set, scores, and lines
            return (scatter_test + scatter + vline + hline)
        
        # Return the combined plot of scores and lines
        return (scatter + vline + hline)


def biplot(scores:pd.DataFrame, 
           loadings:pd.DataFrame,
           comp1:int, 
           comp2:int,
           explained_variance:np.ndarray,
           index_name:str='index',
           hue:pd.Series=None, 
           test_set:pd.DataFrame=None):
    '''
    Generates a scatter plot of the selected components with the scores and the loadings

    Parameters
    ----------
    comp1 : int
        The number of the first component.
    comp2 : int
        The number of the second component.
    hue : pd.Series
        A pandas Series with the hue of the plot. It must have the same length as the number of observations
    test_set : pd.DataFrame
        A pandas DataFrame that contains the held-out observations that will be projected onto the latent space

    Returns
    -------
    None
    '''
    # Create a copy of the scores DataFrame to avoid modifying the original data
    scores = scores.copy()

    # If the number of rows in scores exceeds 5000, randomly sample 5000 rows
    if scores.shape[0] > 5000:
        mask = np.random.choice(scores.shape[0], 5000, replace=False)
        scores = scores.iloc[mask]

    # Calculate the hypothenuse of the scores to scale the loadings
    max_pc1 = scores[f'PC_{comp1}'].max()
    max_pc2 = scores[f'PC_{comp2}'].max()
    hypothenuse = np.sqrt(max_pc1**2 + max_pc2**2)

    # Calculate the hypothenuse of the loadings to scale them
    max_loadings1 = loadings[f'PC_{comp1}'].max()
    max_loadings2 = loadings[f'PC_{comp2}'].max()
    hypothenuse_loadings = np.sqrt(max_loadings1**2 + max_loadings2**2)

    # Calculate the ratio to scale the loadings
    ratio = hypothenuse / hypothenuse_loadings

    # Scale the loadings
    loadings = loadings.copy() * ratio
    loadings.index.name = 'variable'
    loadings.reset_index(inplace=True)

    # Check if hue is provided
    if hue is not None:
        # Ensure hue is a pandas Series
        if not isinstance(hue, pd.Series):
            raise ValueError("Hue must be a pandas Series")
        
        scores[hue.name] = hue

        scores_plot = alt.Chart(scores.reset_index()).mark_circle().encode(
            x=alt.X(f'PC_{comp1}',title=f'PC {comp1} - {explained_variance[comp1-1]*100:.2f} %'),
            y=alt.Y(f'PC_{comp2}',title=f'PC {comp2} - {explained_variance[comp2-1]*100:.2f} %'),
            tooltip=[index_name, f"PC_{comp1}", f"PC_{comp2}", hue.name],
            color=alt.Color(hue.name)
        ).interactive()

    else:
        scores_plot = alt.Chart(scores.reset_index()).mark_circle().encode(
            x=alt.X(f'PC_{comp1}',title=f'PC {comp1} - {explained_variance[comp1-1]*100:.2f} %'),
            y=alt.Y(f'PC_{comp2}',title=f'PC {comp2} - {explained_variance[comp2-1]*100:.2f} %'),
            tooltip=[index_name, f"PC_{comp1}", f"PC_{comp2}"]
        ).interactive()

    
    loadings_plot = alt.Chart(loadings).mark_circle(color='red').encode(
        x=f"PC_{comp1}",
        y=f"PC_{comp2}",
        tooltip=['variable', f"PC_{comp1}", f"PC_{comp2}"]
    )

    hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[12, 6]).encode(y='y')
    vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[12, 6]).encode(x='x')

    if test_set is not None:

        scores_test = test_set.reset_index()
        scatter_test = alt.Chart(scores_test).mark_point(color='black', opacity=.1).encode(
            x=f"PC_{comp1}",
            y=f"PC_{comp2}",
            tooltip=[index_name, f"PC_{comp1}", f"PC_{comp2}"]
        ).interactive()

        return (scatter_test + scores_plot + loadings_plot + vline + hline)

    return (scores_plot + loadings_plot+ vline + hline)

def loadings_barplot(loadings: pd.DataFrame, explained_variance: np.ndarray, comp: int):
    '''
    Generates a bar plot of the loadings of the selected component

    Parameters
    ----------
    loadings : pd.DataFrame
        DataFrame containing the loadings of the variables.
    explained_variance : np.ndarray
        Array containing the explained variance of each component.
    comp : int
        The number of the component.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the bar plot of the loadings.
    '''
    if comp <= 0 or comp > len(explained_variance):
        raise ValueError("Component index is out of range")

    loadings = loadings.copy()
    loadings.index.name = 'variable'
    loadings.reset_index(inplace=True)

    # Altair plot for the loadings
    return alt.Chart(loadings).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y(f'PC_{comp}', title=f'Loadings of PC {comp} - {explained_variance[comp-1]*100:.2f} %'),
        tooltip=['variable', f'PC_{comp}']
    ).interactive()

def hotelling_t2_plot_p1(hotelling: np.ndarray, alpha: float, threshold: float):
    '''
    Generates an interactive plot visualizing the Hotelling's T2 statistic over observations.

    Parameters
    ----------
    hotelling : np.ndarray
        Array containing the Hotelling's T2 statistics for each observation.
    alpha : float
        The significance level used to calculate the control limits.
    threshold : float
        The threshold value for the Hotelling's T2 statistic.

    Returns
    -------
    alt.LayerChart
        An Altair LayerChart object that combines the line plot of Hotelling's T2 statistics with the threshold rule.
    '''
    hotelling_df = pd.DataFrame({'observation': range(len(hotelling)), 'T2': hotelling})

    hotelling_chart = alt.Chart(hotelling_df).mark_line().encode(
        x=alt.X('observation', title='Observation'),
        y=alt.Y('T2', title="Hotelling's T2"),
        tooltip=['observation', "T2"],
    ).properties(
        title=f'Hotelling\'s T2 statistic plot (Phase I) \n alpha: {alpha*100}% -- Threshold: {threshold:.2f}',
    ).interactive()

    hotelling_chart.configure_title(
        fontSize=20,
        font='Courier',
        anchor='start',
        color='gray'
    )

    threshold_line = alt.Chart(
        pd.DataFrame({'y': [threshold]})).mark_rule(
        strokeDash=[12, 6], color='red').encode(y='y')

    # Altair plot for the Hotelling's T2 statistic
    return (hotelling_chart + threshold_line)

def hotelling_t2_plot_p2(hotelling: np.ndarray, alpha: float, hotelling_limit_p2: float) -> alt.LayerChart:
    """
    Generates an interactive plot of the Hotelling's T2 statistic for Phase II observations.

    This function visualizes the Hotelling's T2 statistic for a given set of observations, aiding in the identification of outliers. 
    It creates an interactive line plot with a threshold line indicating the outlier cutoff.

    Parameters
    ----------
    hotelling : np.ndarray
        Array containing the Hotelling's T2 statistics for each observation.
    alpha : float
        The significance level used to calculate the control limits.
    hotelling_limit_p2 : float
        The threshold value for the Hotelling's T2 statistic in Phase II.

    Returns
    -------
    alt.LayerChart
        An Altair LayerChart object that combines the line plot of Hotelling's T2 statistics with the threshold rule.

    Notes
    -----
    - The plot is interactive, allowing for zooming and panning to explore the data points in detail.
    - Observations and their corresponding T2 values are displayed as tooltips when hovering over the plot.
    """
    n_obs = len(hotelling)

    hotelling_df = pd.DataFrame({'observation': range(n_obs), 'T2': hotelling})

    hotelling_chart = alt.Chart(hotelling_df).mark_line().encode(
        x=alt.X('observation', title='Observation'),
        y=alt.Y('T2', title="Hotelling's T2"),
        tooltip=['observation', "T2"],
    ).properties(
        title=f'Hotelling\'s T2 statistic plot (Phase II) \n alpha: {alpha*100}% -- Threshold: {hotelling_limit_p2:.2f}',
    ).interactive()

    hotelling_chart.configure_title(
        fontSize=20,
        font='Courier',
        anchor='start',
        color='gray'
    )

    threshold = alt.Chart(
                    pd.DataFrame({'y': [hotelling_limit_p2]})).mark_rule(
                    strokeDash=[12, 6], color='red').encode(y='y')

    # Altair plot for the Hotelling's T2 statistic
    return (hotelling_chart + threshold)

def spe_plot_p1(spe: np.ndarray, alpha: float, spe_limit: float) -> alt.LayerChart:
    """
    Generates an interactive plot visualizing the Squared Prediction Error (SPE) statistic for Phase I observations.

    This function creates an interactive line plot of the SPE statistic for each observation in the dataset. 
    The plot includes a horizontal dashed line indicating the threshold value beyond which an observation is considered an outlier.

    Parameters
    ----------
    spe : np.ndarray
        Array containing the SPE statistics for each observation.
    alpha : float
        The significance level used to calculate the control limits.
    spe_limit : float
        The threshold value for the SPE statistic.

    Returns
    -------
    alt.LayerChart
        An Altair LayerChart object that combines the line plot of SPE statistics with the threshold rule.

    Notes
    -----
    - The plot is interactive, allowing for zooming and panning to explore the data points in detail.
    - Observations and their corresponding SPE values are displayed as tooltips when hovering over the plot.
    """

    spe_chart = alt.Chart(spe).mark_line().encode(
        x=alt.X('observation', title='Observation'),
        y=alt.Y('SPE', title='SPE'),
        tooltip=['observation', "SPE"],
    ).properties(
        title=f'SPE statistic plot (Phase I) \n alpha: {alpha*100}% -- Threshold: {spe_limit:.2f}',
    ).interactive()

    spe_chart.configure_title(
        fontSize=20,
        font='Courier',
        anchor='start',
        color='gray'
    )

    threshold = alt.Chart(
                    pd.DataFrame({'y': [spe_limit]})).mark_rule(
                    strokeDash=[12, 6], color='red').encode(y='y')

    # Altair plot for the SPE statistic
    return (spe_chart + threshold)

def spe_plot_p2(spe: pd.DataFrame, alpha: float, spe_limit: float) -> alt.LayerChart:
    """
    Generates an interactive plot visualizing the Squared Prediction Error (SPE) statistic for Phase II observations.

    This function creates an interactive line plot of the SPE statistic for each observation in the dataset. 
    The plot includes a horizontal dashed line indicating the threshold value beyond which an observation is considered an outlier.

    Parameters
    ----------
    spe : pd.DataFrame
        DataFrame containing the SPE statistics for each observation. It should have columns 'observation' and 'SPE'.
    alpha : float
        The significance level used to calculate the control limits.
    spe_limit : float
        The threshold value for the SPE statistic.

    Returns
    -------
    alt.LayerChart
        An Altair LayerChart object that combines the line plot of SPE statistics with the threshold rule.

    Notes
    -----
    - The plot is interactive, allowing for zooming and panning to explore the data points in detail.
    - Observations and their corresponding SPE values are displayed as tooltips when hovering over the plot.
    """
    spe_chart = alt.Chart(spe).mark_line().encode(
        x=alt.X('observation', title='Observation'),
        y=alt.Y('SPE', title='SPE'),
        tooltip=['observation', "SPE"],
    ).properties(
        title=f'SPE statistic plot (Phase II) \n alpha: {alpha*100}% -- Threshold: {spe_limit:.2f}',
    ).interactive()

    spe_chart.configure_title(
        fontSize=20,
        font='Courier',
        anchor='start',
        color='gray'
    )

    threshold = alt.Chart(
                    pd.DataFrame({'y': [spe_limit]})).mark_rule(
                    strokeDash=[12, 6], color='red').encode(y='y')

    # Altair plot for the SPE statistic
    return (spe_chart + threshold)

def residuals_barplot(residuals: pd.DataFrame, SPE: np.ndarray, data: pd.DataFrame) -> alt.Chart:
    """
    Generates an interactive bar plot visualizing the residuals for a specific observation within the dataset.

    This function creates a bar plot to display the residuals (differences between observed and predicted values) for a single observation in the dataset. 
    It is designed to help in diagnosing and understanding the prediction errors for individual observations. 
    The plot includes a title indicating the observation index and its Squared Prediction Error (SPE) value.

    Parameters
    ----------
    residuals : pd.DataFrame
        DataFrame containing the residuals for each variable. It should have columns 'variable' and 'residual'.
    SPE : np.ndarray
        Array containing the SPE statistic for the observation.
    data : pd.DataFrame
        The original data for the observation. It is used to extract the observation index for the plot title.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the bar plot of the residuals.

    Notes
    -----
    - The plot is interactive, allowing for tooltips to display the variable names and residual values when hovering over the bars.
    - The title of the plot includes the observation index and its SPE value for reference.
    """
    # Altair plot for the residuals
    return alt.Chart(residuals).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y('residual', title='Residual'),
        tooltip=['variable', 'residual']
    ).properties(
        title=f'Residuals for observation {str(data.index.values[0])} - SPE: {SPE[0]:.2f}'
    ).interactive()

def difference_plot(df_plot: pd.DataFrame) -> alt.Chart:
    """
    Generates an interactive bar plot visualizing the difference between a specific observation and the mean of the sample.

    This function creates a bar plot to visually compare the values of a given observation against the mean values of the training sample. 
    It is particularly useful for understanding how an individual observation deviates from the average trend across each variable. 
    The plot includes tooltips for detailed residual values and is titled with the observation's index and its SPE value, providing immediate insight into the model's performance on that observation.

    Parameters
    ----------
    df_plot : pd.DataFrame
        DataFrame containing the variables and their corresponding values for the observation. 
        It should have columns 'variable' and 'value'.

    Returns
    -------
    alt.Chart
        An interactive Altair Chart object representing the bar plot. This plot includes tooltips and is titled with the observation's index and SPE value.

    Notes
    -----
    - The plot is interactive, allowing for tooltips to display the variable names and values when hovering over the bars.
    - The title of the plot includes the observation index and its SPE value for reference.
    """
    return alt.Chart(df_plot).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y('value', title='Difference with respect to the mean (std)'),
        tooltip=['variable', 'value']
    ).interactive()

import altair as alt
import pandas as pd

def contribution_plot(contributions_df: pd.DataFrame, value: float, obs_name: str, title_prefix: str) -> alt.Chart:
    """
    Generates an interactive bar plot visualizing each variable's contribution to a specific statistic for a specific observation.

    This function creates a bar plot that breaks down the contribution of each variable to the overall value of a given observation. It is useful for identifying which variables contribute most to the observation's deviation from the model's predictions.

    Parameters
    ----------
    contributions_df : pd.DataFrame
        DataFrame containing the contributions of each variable.
    value : float
        The value of the statistic for the observation.
    obs_name : str
        The name or index of the observation being analyzed.
    title_prefix : str
        The prefix for the plot title, indicating the type of statistic (e.g., "SPE", "Hotelling's T2").

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the interactive bar plot of variable contributions.
    pd.DataFrame
        The DataFrame containing the contributions of each variable.

    Notes
    -----
    - The plot includes tooltips for each variable's contribution and displays the total value in the title.
    """
    return alt.Chart(contributions_df).mark_bar().encode(
        x=alt.X('variable', title='Variable'),
        y=alt.Y('contribution', title='Contribution'),
        tooltip=['variable', 'contribution']
    ).properties(
        title=f'Contribution to the {title_prefix} for the observation: {str(obs_name)} - {title_prefix}: {value[0]:.2f}'
    ).interactive(), contributions_df

def spe_contribution_plot(contributions_df: pd.DataFrame, SPE: float, obs_name: str) -> alt.Chart:
    """
    Generates an interactive bar plot visualizing each variable's contribution to the Squared Prediction Error (SPE) for a specific observation.

    Parameters
    ----------
    contributions_df : pd.DataFrame
        DataFrame containing the contributions of each variable to the SPE.
    SPE : float
        The Squared Prediction Error (SPE) value for the observation.
    obs_name : str
        The name or index of the observation being analyzed.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the interactive bar plot of variable contributions to the SPE.
    pd.DataFrame
        The DataFrame containing the contributions of each variable to the SPE.
    """
    return contribution_plot(contributions_df, SPE, obs_name, "SPE")

def hotelling_t2_contribution_plot(contributions_df: pd.DataFrame, hotelling: float, obs_name: str) -> alt.Chart:
    """
    Generates an interactive bar plot visualizing each variable's contribution to the Hotelling's T2 statistic for a specific observation.

    Parameters
    ----------
    contributions_df : pd.DataFrame
        DataFrame containing the contributions of each variable to the Hotelling's T2 statistic.
    hotelling : float
        The Hotelling's T2 value for the observation.
    obs_name : str
        The name or index of the observation being analyzed.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the interactive bar plot of variable contributions to the Hotelling's T2 statistic.
    pd.DataFrame
        The DataFrame containing the contributions of each variable to the Hotelling's T2 statistic.
    """
    return contribution_plot(contributions_df, hotelling, obs_name, "Hotelling's T2")