import pandas as pd

def preprocess(data: pd.DataFrame, scaler, numerical_features: list) -> pd.DataFrame:
    """
    Preprocesses the input data using the provided scaler.

    This function applies scaling to the numerical features of the input data. If specific numerical features are 
    defined, only those features are scaled. Otherwise, scaling is applied to all features. The function ensures 
    that the original index and column names of the input data are preserved in the transformed output.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to preprocess. Must be a pandas DataFrame.
    scaler : sklearn.pipeline.Pipeline
        The scaler pipeline to use for preprocessing.
    numerical_features : list
        List of numerical features to scale. If empty, all features are scaled.

    Returns
    -------
    pd.DataFrame
        The scaled version of the input data, with the original index and columns preserved.
    """
    index = data.index
    columns = data.columns

    X_transform = data.copy()

    if numerical_features:
        X_transform[numerical_features] = scaler.transform(X_transform[numerical_features])
    else:
        X_transform = pd.DataFrame(scaler.transform(X_transform), columns=columns, index=index)

    return X_transform