class NotDataFrameError(TypeError):
    """
    Exception raised when the input data is not a pandas DataFrame.
    """
    def __init__(self, input_type):
        message = f"Data must be of type pandas DataFrame, not {input_type}."
        super().__init__(message)

class ModelNotFittedError(ValueError):
    """
    Exception raised when attempting to use a model's method that requires the model to be fitted, but the model is not.
    """
    def __init__(self):
        message = "The model has not been fitted yet. Please use the fit method before predicting."
        super().__init__(message)

class NotAListError(TypeError):
    """
    Exception raised when the input data is not a list
    """
    def __init__(self, input_type):
        message = f"Data must be of type list, not {input_type}"
        super().__init__(message)

class NotBoolError(TypeError):
    """
    Exception raised when a boolean parameter receives a different type
    """
    def __init__(self):
        message = f"Data must be of type bool"
        super().__init__(message)

class NComponentsError(ValueError):
    """
    Exception raised when the number of components is invalid
    """
    def __init__(self, data_vars):
        message = f'The number of components must be between 0 and {data_vars}'
        super().__init__(message)
