
import functools
import pandas as pd
from .exceptions import NotDataFrameError, ModelNotFittedError

def validate_dataframe(param_name: str = "data"):
    """
    Decorator that validates if the specified parameter is a pandas DataFrame.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Try to find the parameter by name (from kwargs) or by position in args
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            value = bound_args.arguments.get(param_name)

            if not isinstance(value, pd.DataFrame):
                raise NotDataFrameError(type(value).__name__)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def require_fitted(method):
    """Decorator to ensure the model is fitted before proceeding."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_fitted():
            raise ModelNotFittedError()
        return method(self, *args, **kwargs)
    return wrapper

def cache_result(method):
    """Decorator to cache the result of methods that are expensive to compute."""
    attr_name = f"_{method.__name__}_cache"
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        result = method(self, *args, **kwargs)
        setattr(self, attr_name, result)
        return result
    return wrapper