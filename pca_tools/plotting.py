import altair as alt
import pandas as pd
import numpy as np
from typing import Union
import re

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

def generic_stat_plot(data: pd.DataFrame,
                      stat_column: str,
                      control_limit: float,
                      alpha: float,
                      phase: str,
                      y_label: str = None) -> alt.LayerChart:
    """
    Generate an interactive Altair plot for a given statistic with a control threshold line.

    This function creates an interactive line plot for the specified statistic over observations.
    It includes a horizontal dashed line representing the control limit and a title that annotates the
    significance level (alpha) and threshold value. The input data can be a pandas DataFrame or a NumPy array.
    If a NumPy array is provided, it is assumed to represent statistic values with the observation index generated automatically.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data containing the statistic values. If a DataFrame is provided, it is expected to have a column
        corresponding to the statistic. If a NumPy array is provided, the array is considered as the statistic values.
    stat_column : str
        The name of the column representing the statistic (e.g., 'T2', 'SPE', 'DModX').
    control_limit : float
        The control threshold for the statistic.
    alpha : float
        The significance level used to compute the control limits.
    phase : str
        Label indicating the phase (e.g., "Phase I" or "Phase II").
    y_label : str, optional
        Label for the y-axis. Defaults to the value of stat_column if not provided.

    Returns
    -------
    alt.LayerChart
        An Altair LayerChart object that overlays the statistic line plot with a horizontal threshold rule.
    """
    # Ensure the input data is a DataFrame.
    df = data.copy()
    if 'observation' not in df.columns:
        df['observation'] = range(len(df))
    
    if y_label is None:
        y_label = stat_column

    # Format the plot title.
    title = f"{stat_column} plot ({phase}) \n alpha: {alpha * 100}% -- Threshold: {control_limit:.2f}"

    # Create the main line chart (without any individual config).
    stat_chart = alt.Chart(df).mark_line().encode(
        x=alt.X('observation', title='Observation'),
        y=alt.Y(f'{stat_column}', title=y_label),
        tooltip=[alt.Tooltip('observation', title='Observation'),
                 alt.Tooltip(f'{stat_column}', title=stat_column)]
    ).properties(
        title=title
    ).interactive()

    stat_chart.configure_title(
        fontSize=20,
        font='Courier',
        anchor='start',
        color='gray'
    )

    # Create the threshold rule.
    threshold_line = alt.Chart(pd.DataFrame({'y': [control_limit]})).mark_rule(
        strokeDash=[12, 6],
        color='red'
    ).encode(
        y=alt.Y('y:Q')
    )

    # # Combine the charts using alt.layer and define the title configuration for the layered chart.
    # layered_chart = alt.layer(stat_chart, threshold_line).configure_title(
    #     fontSize=20,
    #     font='Courier',
    #     anchor='start',
    #     color='gray'
    # )

    return (stat_chart + threshold_line)


def residuals_barplot(residuals: pd.DataFrame, SPE: np.ndarray, obs_name:str) -> alt.Chart:
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
        title=f'Residuals for observation {str(obs_name)} - SPE: {SPE[0]:.2f}'
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

def dmodx_contribution_plot(contributions_df: pd.DataFrame, dmodx: float, obs_name: str) -> alt.Chart:
    """
    Generates an interactive bar plot visualizing each variable's contribution to the DModX statistic for a specific observation.

    Parameters
    ----------
    contributions_df : pd.DataFrame
        DataFrame containing the contributions of each variable to the DModX statistic.
    dmodx : float
        The DModX value for the observation.
    obs_name : str
        The name or index of the observation being analyzed.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the interactive bar plot of variable contributions to the DModX statistic.
    pd.DataFrame
        The DataFrame containing the contributions of each variable to the DModX statistic.
    """
    return contribution_plot(contributions_df, dmodx, obs_name, "DModX")

def actual_vs_predicted(predictions_df: pd.DataFrame) -> alt.Chart:
    """
    Create an interactive grouped bar chart comparing Actual vs Predicted values for each variable.

    This function formats the input DataFrame in long format (with columns 'Variable', 'Type', and 'Value')
    and generates an Altair chart that displays side-by-side bars for each variable. The chart width is
    dynamically adjusted based on the number of unique variables present in the data.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        A DataFrame in long format with the following columns:
            - 'Variable': Name of the variable.
            - 'Type': Category of the value (e.g., 'Actual', 'Predicted').
            - 'Value': Numeric value corresponding to the variable and type.

    Returns
    -------
    alt.Chart
        An interactive Altair chart displaying Actual vs Predicted values per variable.
    """
    # Compute the number of unique variables and set the desired width per variable.
    num_vars = predictions_df['Variable'].nunique()
    width_per_variable = 60  # Adjust the width per variable if needed.
    chart_width = num_vars * width_per_variable

    # Create the chart.
    chart = alt.Chart(predictions_df).mark_bar().encode(
        x=alt.X('Variable:N', title='Variable', axis=alt.Axis(labelAngle=45)),
        xOffset=alt.X('Type:N'),
        y=alt.Y('Value:Q', title='Value'),
        color=alt.Color('Type:N', scale=alt.Scale(scheme='category10')),
        tooltip=[
            alt.Tooltip('Variable:N', title='Variable'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('Value:Q', title='Value', format=".2f")
        ]
    ).properties(
        width=chart_width,
        height=400,
        title='Actual vs Predicted Values per Variable'
    ).interactive()

    return chart

def structural_variance_plot(alpha_A: pd.DataFrame, R2_A: pd.DataFrame, variable_name: str) -> alt.Chart:
    """
    Create an interactive Altair chart displaying the structural variance information for a given variable.
    
    This function extracts the self-explanatory power (alpha_A) for the selected variable and the corresponding R² values,
    merges them into a long-format DataFrame, and creates a line chart with points. The x-axis (Principal Component) is 
    sorted in natural numeric order.
    
    Parameters
    ----------
    alpha_A : pd.DataFrame
        DataFrame containing the self-explanatory power values. Its index should be variable names and columns the PC labels.
    R2_A : pd.DataFrame
        DataFrame containing the R² values for each principal component (with PCs as index).
    variable_name : str
        The name of the variable for which the plot is generated.
    
    Returns
    -------
    alt.Chart
        An interactive Altair chart showing SVI metrics for the specified variable.
    """
    # Extract the self-explanatory power for the given variable and reset the index:
    alpha_series = alpha_A.loc[variable_name].reset_index()
    alpha_series.columns = ['PC', 'alpha_value']
    
    # Reset the index for the R2 DataFrame:
    R2_series = R2_A.reset_index()
    R2_series.columns = ['PC', 'R2_value']

    # Merge the two series on the PC column:
    merged_df = pd.merge(alpha_series, R2_series, on='PC')
    
    # Melt the merged DataFrame into long format:
    melted_df = merged_df.melt(id_vars='PC', 
                               value_vars=['alpha_value', 'R2_value'], 
                               var_name='Metric', value_name='Value')
    
    # Create a custom sort order based on the integer part of "PC".
    def extract_pc_num(pc_label):
        match = re.search(r'\d+', pc_label)
        return int(match.group()) if match else 0

    sorted_pcs = sorted(melted_df['PC'].unique(), key=extract_pc_num)
    
    # Create the Altair chart with custom x-axis sort:
    return alt.Chart(melted_df).mark_line(point=True).encode(
        x=alt.X('PC:N', title='Principal Component', sort=sorted_pcs),
        y=alt.Y('Value:Q', title='Explained variance'),
        color=alt.Color('Metric:N', title='SVI Metric', 
                        scale=alt.Scale(domain=['alpha_value', 'R2_value'],
                                        range=['blue', 'orange'])),
        tooltip=[alt.Tooltip('PC:N', title='Principal Component'),
                 alt.Tooltip('Metric:N', title='Metric'),
                 alt.Tooltip('Value:Q', title='Value', format=".2f")]
    ).properties(
        title=f'Structural and Variance Information (SVI) for {variable_name}',
        width=600,
        height=400
    ).interactive()