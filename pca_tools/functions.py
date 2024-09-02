import pandas as pd

def contribution_status(row):
    """
    Determine the status based on the relative contribution of a variable.

    This function takes a row of a DataFrame and evaluates the 'relative_contribution'
    value to assign a status. The status is determined as follows:
    - 'red' if the relative contribution is greater than 0.5
    - 'yellow' if the relative contribution is greater than 0.3 but less than or equal to 0.5
    - 'green' if the relative contribution is less than or equal to 0.3

    Parameters:
    row (pd.Series): A row from a pandas DataFrame containing at least the 'relative_contribution' column.

    Returns:
    str: The status ('red', 'yellow', or 'green') based on the relative contribution.
    """
    if row['relative_contribution'] > 0.5:
        return 'red'
    elif row['relative_contribution'] > 0.3:
        return 'yellow'
    else:
        return 'green'