from typing import List, Tuple
import numpy as np
import pandas as pd
import logging
from .model import PCA
from .exceptions import NotDataFrameError, NComponentsError, NotAListError

class PCAOptimizer:
    """
    Class to optimize a dataset for PCA by iteratively removing out-of-control observations.

    Attributes
    ----------
    n_comps : int
        The number of principal components to use for PCA.
    alpha : float
        The significance level for computing control limits (between 0 and 1).
    numerical_features : List[str]
        The list of numerical features to be considered in the PCA.
    statistic : str
        The control statistic to use ('T2' for Hotelling's T² or 'SPE' for Squared Prediction Error).
    threshold : float
        The factor multiplying the control limit to flag out-of-control observations.
    drop_percentage : float
        Proportion (between 0 and 1) of the out-of-control observations to drop in each iteration.
    max_iterations : int
        Maximum iterations allowed to avoid infinite loops.
    """
    def __init__(self, n_comps: int, alpha: float, numerical_features: List[str], 
                 statistic: str = 'T2', threshold: float = 3, drop_percentage: float = 0.2,
                 max_iterations: int = 50):
        # Validate basic parameters.
        if statistic not in ['T2', 'SPE']:
            raise ValueError("Statistic must be 'T2' or 'SPE'.")
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1.")
        if threshold <= 0:
            raise ValueError("Threshold must be positive.")
        if not (0 < drop_percentage <= 1):
            raise ValueError("Drop percentage must be between 0 and 1.")
        if not isinstance(numerical_features, list):
            raise NotAListError()
            
        self.n_comps = n_comps
        self.alpha = alpha
        self.numerical_features = numerical_features
        self.statistic = statistic
        self.threshold = threshold
        self.drop_percentage = drop_percentage
        self.max_iterations = max_iterations

        self.logger = logging.getLogger(__name__)

    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate that the input data is a DataFrame and has sufficient columns.
        
        Raises
        ------
        NotDataFrameError
            If X is not a pandas DataFrame.
        NComponentsError
            If n_comps is not in the valid range.
        """
        if not isinstance(X, pd.DataFrame):
            raise NotDataFrameError(type(X).__name__)
        if self.n_comps <= 0 or self.n_comps > X.shape[1]:
            raise NComponentsError(X.shape[1])

    def _fit_pca(self, X: pd.DataFrame) -> PCA:
        """
        Create and fit the PCA model on the provided data.
        
        Returns
        -------
        PCA
            A fitted PCA model.
        """
        pca = PCA(n_comps=self.n_comps, numerical_features=self.numerical_features, alpha=self.alpha)
        pca.fit(X)
        return pca

    def _get_control_limit_and_stats(self, pca: PCA) -> Tuple[float, np.ndarray]:
        """
        Get the control limit and the associated statistic values from the PCA model.
        
        Returns
        -------
        Tuple[float, np.ndarray]
            A tuple containing the control limit and the statistic values.
        """
        if self.statistic == 'T2':
            control_limit = pca._hotelling_limit_p1
            stat_values = np.array(pca._hotelling)
        else:  # SPE
            control_limit = pca._spe_limit
            stat_values = np.array(pca._spe)
        return control_limit, stat_values

    def _get_drop_indices(self, stat_values: np.ndarray, control_limit: float) -> np.ndarray:
        """
        Identify and sort the indices of out-of-control observations to drop.
        
        Parameters
        ----------
        stat_values : np.ndarray
            Array of statistic values (T2 or SPE).
        control_limit : float
            The control limit for the statistic.
        
        Returns
        -------
        np.ndarray
            Sorted indices (in descending order by statistic value) to drop.
        """
        out_indices = np.where(stat_values > control_limit)[0]
        if len(out_indices) == 0:
            return out_indices
        sorted_indices = out_indices[np.argsort(stat_values[out_indices])[::-1]]
        drop_count = max(1, int(len(sorted_indices) * self.drop_percentage))
        return sorted_indices[:drop_count]

    def optimize(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize the dataset by iteratively removing out-of-control observations.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to be optimized.

        Returns
        -------
        pd.DataFrame
            The optimized data with out-of-control observations removed.
        """
        self._validate_input(X)
        # Initialize indices of observations to keep.
        keep_indices = np.arange(X.shape[0])
        iteration = 0

        # Initial fit.
        pca = self._fit_pca(X)
        control_limit, stat_values = self._get_control_limit_and_stats(pca)
        out_of_control = np.sum(stat_values > control_limit) / len(keep_indices)

        # Loop to remove out-of-control observations.
        while out_of_control > (1 - self.alpha) * self.threshold and iteration < self.max_iterations:
            drop_indices = self._get_drop_indices(stat_values, control_limit)
            if len(drop_indices) == 0:
                break
            self.logger.info(f"Iteration {iteration+1}: Removing {len(drop_indices)} out-of-control observations.")
            self.logger.info(f"Control limit: {control_limit:.4f}, Proportion out-of-control: {out_of_control:.4f}")
            keep_indices = np.delete(keep_indices, drop_indices)
            # Retrain PCA with the remaining data.
            pca = self._fit_pca(X.iloc[keep_indices])
            control_limit, stat_values = self._get_control_limit_and_stats(pca)
            out_of_control = np.sum(stat_values > control_limit) / len(keep_indices)
            iteration += 1

        self.logger.info(f"Optimization finished after {iteration} iteration(s).")
        self.logger.info(f"Final proportion out-of-control: {out_of_control:.4f}")
        
        return X.iloc[keep_indices]