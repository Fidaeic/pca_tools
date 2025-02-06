# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:33:50 2020

author: Fidae El Morer
"""
import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.stats import f, beta, chi2
import altair as alt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .exceptions import NotDataFrameError, ModelNotFittedError, NotAListError, NotBoolError, NComponentsError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA as PCA_sk
from .plotting import score_plot, biplot, loadings_barplot, residuals_barplot, spe_contribution_plot, hotelling_t2_contribution_plot, actual_vs_predicted, dmodx_contribution_plot, generic_stat_plot
from .decorators import validate_dataframe, require_fitted, cache_result
from .preprocess import preprocess

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_comps:int=None, 
                 numerical_features:list=[], 
                 standardize:bool=True, 
                 tolerance:float=1e-4, 
                 alpha:float=.99) -> None:
        
        if not 0 < tolerance < 1:
            raise ValueError('Tolerance must be strictly between 0 and 1')
        
        if not 0 < alpha < 1:
            raise ValueError('Alpha must be strictly between 0 and 1')

        if not isinstance(numerical_features, list):
            raise NotAListError(type(numerical_features).__name__)
        
        self._standardize = standardize
        self._tolerance = tolerance
        self._ncomps = n_comps
        self._numerical_features = numerical_features
        self._alpha = alpha
        self.model = PCA_sk(n_components=self._ncomps, svd_solver='full', tol=self._tolerance, iterated_power='auto')

        self._scaler = Pipeline([
            ('scaler', StandardScaler())
        ])

    def get_params(self, deep=True):
        return {
            'n_comps': self._ncomps,
            'numerical_features': self._numerical_features,
            'standardize': self._standardize,
            'tolerance': self._tolerance,
            'alpha': self._alpha
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_rsquared_acc(self):
        return self._rsquared_acc
    
    def is_fitted(self) -> bool:
        """
        Checks if the PCA model has been fitted.
        
        Returns
        -------
        bool
            True if the model has been fitted (i.e., the _scores attribute is present); False otherwise.
        """
        return hasattr(self, '_scores')
    
    def _validate_ncomps(self, data: pd.DataFrame):
        if self._ncomps is None:
            self._ncomps = data.shape[1]

        if self._ncomps <= 0 or self._ncomps > data.shape[1]:
            raise NComponentsError(data.shape[1])

    @validate_dataframe('data')
    def _initialize_attributes(self, data: pd.DataFrame):
        self._variables = data.columns
        self._index = data.index
        self._index_name = data.index.name
        self._nobs, self._nvars = data.shape

    @validate_dataframe('data')
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        X = data.copy()
        if self._standardize:
            X = preprocess(data=X, scaler=self._scaler, numerical_features=self._numerical_features)
        return X.values
    
    def _fit_model(self, X: np.ndarray):
        self.model.fit(X)
        self._loadings = pd.DataFrame(self.model.components_.T, columns=[f"PC_{i+1}" for i in range(self._ncomps)], index=self._variables)
        self._scores = pd.DataFrame(self.model.transform(X), columns=[f"PC_{i+1}" for i in range(self._ncomps)], index=self._index)

    def _calculate_metrics(self, data: pd.DataFrame, X: np.ndarray):
        self._explained_variance = self.model.explained_variance_ratio_
        self._rsquared_acc = np.cumsum(self.model.explained_variance_ratio_)
        self._eigenvals = np.var(self._scores.values, axis=0)
        self._residuals_fit = X - self._scores @ self._loadings.T
        self._mean_train = np.mean(data.values, axis=0)
        self._std_train = np.std(data.values, axis=0)

    @validate_dataframe('data')
    def fit(self, data, y=None):
        """
        Fit the PCA model using the provided training data.
    
        If standardization is enabled (self._standardize), the method will:
          - Determine the numerical features from the input data (if not explicitly provided via self._numerical_features).
          - Fit a scaler on these numerical features.
        
        The method then trains the PCA model by calling self.train(data), and calculates model performance metrics, including the Squared Prediction Error (SPE) and Hotelling's T² statistic for the training set.
        Finally, the control limits for Hotelling's T² (Phase I and Phase II) and SPE are computed using the specified significance level (alpha).
    
        Parameters
        ----------
        data : pd.DataFrame
            The input data used for training the PCA model. It should contain the features intended for analysis.
        y : Any, optional
            Not used in this PCA model. Included for compatibility with scikit-learn's estimator API.
    
        Returns
        -------
        self
            Returns the fitted instance of the PCA model.
        """
        if self._standardize:
            if not self._numerical_features:
                self._numerical_features = data.columns.tolist()
            
            self._scaler.fit(data[self._numerical_features])
    
        self.train(data)
    
        self._spe, _ = self.spe(data)
        self._hotelling = self.hotelling_t2(data)
        self._dmodx = self.dmodx(data)
    
        self._hotelling_limit_p1, self._hotelling_limit_p2, self._spe_limit, self._dmodx_limit = self.control_limits(alpha=self._alpha)
        
        return self

    @validate_dataframe('data')
    def preprocess(self, data):
        """
        Preprocesses the input data using the scaler attribute of the class.

        This method applies scaling to the numerical features of the input data. If specific numerical features are 
        defined in the class (`self._numerical_features`), only those features are scaled. Otherwise, scaling is applied 
        to all features. The method ensures that the original index and column names of the input data are preserved 
        in the transformed output.

        Parameters:
        - data (pd.DataFrame): The input data to preprocess. Must be a pandas DataFrame.

        Returns:
        - X_transform (pd.DataFrame): The scaled version of the input data, with the original index and columns preserved.
        """
        return preprocess(data, self._scaler, self._numerical_features)
    
    @validate_dataframe('data')
    def train(self, data: pd.DataFrame):
        '''
        Trains the PCA model using the provided dataset.

        This method fits the Principal Component Analysis (PCA) model to the given dataset, extracting the principal components up to the number specified during the model's initialization. It performs several key steps, including validating the input data, preprocessing it if necessary, and then iteratively extracting each principal component until the specified number of components is reached or the variance explained by the components meets a predefined threshold.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to fit the PCA model to, where rows represent samples and columns represent features.

        Raises
        ------
        NComponentsError
            If the number of components specified is not within the valid range (1 to the number of features in `data`).
        NotDataFrameError
            If the input `data` is not a pandas DataFrame.

        Returns
        -------
        None
            This method updates the PCA model in-place, setting various attributes related to the fitted model, including the eigenvalues, loadings, scores, and the proportion of variance explained by each principal component.

        Notes
        -----
        - The method assumes `data` is a pandas DataFrame with numeric columns.
        - If the number of components (`_ncomps`) is not set, it defaults to the number of features in `data`.
        - The method preprocesses `data` by descaling and demeaning if `_standardize` is True.
        - Principal components are extracted using an iterative process that maximizes the variance explained by each component.
        - The method calculates and stores various metrics related to the PCA model, including the loadings, scores, eigenvalues, and the cumulative variance explained by the extracted components.
        '''
        self._validate_ncomps(data)
        self._initialize_attributes(data)
        X = self._preprocess_data(data)
        self._fit_model(X)
        self._calculate_metrics(data, X)

    @validate_dataframe('data')
    @require_fitted
    def transform(self, data:pd.DataFrame, y=None):
        '''
        Transforms the input data by projecting it onto the PCA space defined by the model.

        This method takes a dataset and projects it onto the PCA space, effectively reducing its dimensionality based on the principal components previously extracted during the model's fitting process. It is designed to work seamlessly within a scikit-learn pipeline, hence the optional `y` parameter, which is not used but ensures compatibility with pipeline requirements.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be transformed, where each row is a sample and each column is a feature. The dataset must be a pandas DataFrame.
        y : None, optional
            Placeholder to maintain compatibility with scikit-learn's pipeline transformations. It is not used in the transformation process.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the transformed data. The DataFrame has the same number of rows as the input data and a number of columns equal to the number of principal components retained by the model. The columns are named according to the principal components they represent.

        Raises
        ------
        NotDataFrameError
            If the input `data` is not a pandas DataFrame.

        Notes
        -----
        - The transformation process involves centering and scaling the data (if standardization was applied during fitting) before projecting it onto the PCA space using the loadings matrix derived during fitting.
        - The returned DataFrame retains the original index of the input `data`, facilitating easy tracking of samples.
        '''
        X_transform = data.copy()
        # Descale and demean matrix
        if self._standardize:
            X_transform = self.preprocess(data=X_transform)
        
        return pd.DataFrame(X_transform @ self._loadings, columns=self._scores.columns, index=data.index)

    @validate_dataframe('data')
    def fit_transform(self, data, y=None):
        '''
        Fits the PCA model to the data and then transforms the data by projecting it onto the PCA space.

        This method first fits the PCA model to the provided dataset, identifying the principal components that capture the most variance within the data. After fitting, it transforms the dataset by projecting it onto the space spanned by the identified principal components. This process reduces the dimensionality of the data while attempting to preserve as much of the data's variance as possible.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be transformed, where each row is a sample and each column is a feature. The dataset must be a pandas DataFrame.
        y : None, optional
            Included for compatibility with scikit-learn pipeline conventions. It is not used in this method.

        Returns
        -------
        pd.DataFrame, shape (n_samples, n_components)
            The data transformed into the PCA space, with dimensionality reduced to the number of principal components retained by the model.

        Notes
        -----
        - The method combines fitting and transformation into a single step, which is particularly useful for pipeline integrations where model fitting and data transformation are performed sequentially.
        - The number of principal components (`n_components`) retained can be specified during the model initialization. If not specified, it defaults to the lesser of the number of samples or features in `data`.
        '''
        self.train(data)
        return self.transform(data)

    @require_fitted
    @validate_dataframe('data')
    def inverse_transform(self, data:pd.DataFrame):
        '''
        Reconstructs the original dataset from its PCA-transformed version.

        This method takes a dataset that has been transformed into the PCA space and projects it back to its original feature space. This is achieved by multiplying the PCA-transformed data by the transpose of the loadings matrix, effectively reversing the PCA transformation process.

        Parameters
        ----------
        data : pd.DataFrame, shape (n_samples, n_components)
            The PCA-transformed data to be projected back to the original space. Can be a NumPy array or a pandas DataFrame.

        Returns
        -------
        pd.DataFrame, shape (n_samples, n_features)
            The data reconstructed in the original feature space, with the same number of features as the original dataset before PCA transformation.

        Notes
        -----
        - This operation is the inverse of the PCA transformation, but it may not perfectly reconstruct the original data if the PCA transformation was lossy (i.e., if some components were discarded).
        '''
        # Reconstruct in the scaled space via PCA loadings multiplication.
        # Ensure data is converted to NumPy array for multiplication.
        projection = data.values @ self._loadings.values.T

        if self._standardize:
            # Create a DataFrame for clarity and to align columns.
            reconstructed_df = pd.DataFrame(projection, index=data.index, columns=self._variables)
            
            if self._numerical_features:
                # Apply inverse scaling only on the numerical features.
                # This implies that only the numerical columns were scaled during preprocessing.
                scaled_part = reconstructed_df[self._numerical_features]
                inv_scaled = self._scaler.inverse_transform(scaled_part)
                # Replace the scaled numerical features with their inverse-scaled values.
                reconstructed_df.loc[:, self._numerical_features] = inv_scaled
                result = reconstructed_df.values
            else:
                # If no specific numerical features were provided, inverse scale the entire projection.
                result = self._scaler.inverse_transform(projection)
        else:
            result = projection

        return pd.DataFrame(result, columns=self._variables, index=data.index)
    
    @cache_result
    @require_fitted
    def control_limits(self, alpha: float = 0.95) -> tuple[float, float, float]:
        """
        Calculates the control limits for Hotelling's T2, SPE (Squared Prediction Error), and DModX statistics.
    
        This method computes the control limits for both Hotelling's T2 and SPE statistics for a PCA model.
        Two phases of control limits are computed:
          - Phase I: Based on the model fitting dataset.
          - Phase II: For monitoring new observations.
        These limits help detect outliers or shifts in process variability.
    
        Parameters
        ----------
        alpha : float, optional
            The significance level used to calculate the control limits (default is 0.95, corresponding to a 95% confidence level).
    
        Returns
        -------
        tuple[float, float, float]
            A tuple containing:
                - hotelling_limit_p1 : float
                    Control limit for Hotelling's T2 in Phase I.
                - hotelling_limit_p2 : float
                    Control limit for Hotelling's T2 in Phase II.
                - spe_limit : float
                    Control limit for the SPE statistic.
                - dmodx_limit : float
                    Control limit for the DMODX statistic.
    
        Notes
        -----
        - Hotelling's T2 is a multivariate analogue of the univariate t-square statistic.
        - SPE measures the squared prediction error from the PCA model.
        - DModX is a measure of the absolute distance to the model
        - Limits are derived from the beta, F-, and chi-squared distributions, adjusted for sample size and number of components.
        """
        # Phase I: Hotelling's T2 control limit
        # Degrees of freedom for beta distribution approximation
        dfn_phase1 = self._ncomps / 2
        dfd_phase1 = (self._nobs - self._ncomps - 1) / 2
        constant_phase1 = ((self._nobs - 1) ** 2) / self._nobs
        hotelling_limit_p1 = beta.ppf(alpha, dfn_phase1, dfd_phase1) * constant_phase1
    
        # Phase II: Hotelling's T2 control limit
        constant_phase2 = (self._ncomps * (self._nobs ** 2 - 1)) / (self._nobs * (self._nobs - self._ncomps))
        dfn_phase2 = self._ncomps
        dfd_phase2 = self._nobs - self._ncomps
        hotelling_limit_p2 = f.ppf(alpha, dfn_phase2, dfd_phase2) * constant_phase2
    
        # SPE (Squared Prediction Error) control limit
        mean_spe = np.mean(self._spe)
        var_spe = np.var(self._spe)
        # Calculate degrees of freedom based on SPE moments
        df_spe = (2 * mean_spe ** 2) / var_spe if var_spe != 0 else 1  # safeguard against division by zero
        constant_spe = var_spe / (2 * mean_spe) if mean_spe != 0 else 1
        spe_limit = chi2.ppf(alpha, df_spe) * constant_spe

        # DMODX control limit
        m = self._nobs
        K = self._nvars
        A = self._ncomps

        # Pooled residual standard deviation
        s_0 = np.sqrt(np.sum(self._residuals_fit.values**2)/((m-A-1)*(K-A)))

        dfn = K-A
        dfd = (m-A-1)*(K-A)
        dmodx_limit = s_0*np.sqrt(f.ppf(alpha, dfn, dfd))
    
        return hotelling_limit_p1, hotelling_limit_p2, spe_limit, dmodx_limit
    
    @validate_dataframe('data')
    @require_fitted
    def hotelling_t2(self, data: pd.DataFrame) -> list[float]:
        """
        Compute Hotelling's T² statistic for each observation.
    
        Hotelling's T² represents the estimated squared Mahalanobis distance from the center of the latent subspace to the projection of each observation. It is calculated as:
        
            T² = sum((t_a)**2 / lambda_a)
        
        where lambda_a is the eigenvalue of the ath component and t_a is the score of the observation on the ath component. Under the assumption of multivariate normal scores, T² can be used to detect outliers; observations with T² above the control limit may be considered extreme outliers.
    
        Parameters
        ----------
        data : pd.DataFrame
            Input data to be transformed onto the latent space.
    
        Returns
        -------
        list[float]
            A list of Hotelling's T² statistics, one per observation.
        """
        # Project data using the fitted PCA model.
        predicted_scores = self.transform(data)
        
        # Calculate Hotelling's T²: square the scores and normalize by corresponding eigenvalues.
        # Using np.square for clarity.
        t2_values = np.sum(np.square(predicted_scores.values) / self._eigenvals, axis=1)
        
        # Returning the result as a list for consistency with previous implementations.
        return t2_values.tolist()
    
    @validate_dataframe('data')
    @require_fitted
    def spe(self, data: pd.DataFrame) -> tuple[list[float], np.ndarray]:
        """
        Computes the Squared Prediction Error (SPE) statistic for every observation in the given data.
    
        SPE is defined as the squared Euclidean distance between the original data and its projection onto the PCA subspace. In other words, for each observation, SPE is calculated as:
        
            SPE = || e ||²  where e = original observation - reconstruction
    
        High SPE values indicate that the observation is not well represented by the PCA model, which may suggest that the observation is an outlier.
    
        Parameters
        ----------
        data : pd.DataFrame
            The input data for which the SPE statistic will be calculated.
    
        Returns
        -------
        tuple[list[float], np.ndarray]
            A tuple containing:
                - A list of SPE values for each observation.
                - A NumPy array of residuals (differences between the original data and its reconstruction).
        """
        # Create a copy of the data to avoid modifying the original DataFrame.
        X_transform = data.copy()
    
        # Reconstruct the data from the PCA scores by multiplying with the transpose of the loadings matrix.
        reconstruction = self.inverse_transform(self.transform(data))
    
        # Calculate the residuals: difference between original data and its reconstruction.
        residuals = X_transform - reconstruction
    
        # Calculate the SPE for each observation as the sum of squared residuals.
        SPE_values = np.sum(np.square(residuals), axis=1)
    
        return SPE_values.tolist(), residuals
    
    @validate_dataframe('data')
    @require_fitted
    def project(self, data:pd.DataFrame):
        """
        Projects new data onto the fitted PCA model and calculates Hotelling's T2 and SPE statistics.

        This method is used to project new observations onto the principal components obtained from the fitted PCA model. 
        It also calculates the Hotelling's T2 statistic and the Squared Prediction Error (SPE) for each observation, 
        which can be used for anomaly detection or assessing the fit of the model.

        Parameters:
        - data (pd.DataFrame): The new data to be projected. Must be a pandas DataFrame.

        Raises:
        - ValueError: If `data` is not a pandas DataFrame.
        - ValueError: If the model has not been fitted yet.

        Returns:
        - hotelling_p2 (list of float): The Hotelling's T2 statistic for each observation in `data`.
        - spe_p2 (list of float): The SPE statistic for each observation in `data`.
        - residuals (np.ndarray): The residuals of the projection, indicating the difference between the original 
        data and its reconstruction from the principal components.
        - predicted_scores (pd.DataFrame): The scores of the projected data on the principal components.
        """
        
        hotelling_p2 = self.hotelling_t2(data)
        
        predicted_scores = self.transform(data)

        spe_p2, residuals = self.spe(data)

        return hotelling_p2, spe_p2, residuals, predicted_scores
    
    @validate_dataframe('data')
    @require_fitted
    def predict(self, data:pd.DataFrame):
        '''
        Predicts the probability of an observation being an outlier

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            The data to be predicted
        Returns
        -------
        response : dict
            Dictionary containing the prediction of the model given an observation. It contains attributes such as the T2 and SPE limits and values, as well as the probability of the observation being an outlier
        '''
        hotelling, SPE, _, _ = self.project(data)

        X_transform = data.copy()
        if self._standardize:
            X_transform = self.preprocess(data=X_transform)

        t2_contributions, _ = self.t2_contribution(data)
        spe_contributions, _ = self.spe_contribution(data)

        if data.shape[1]>1:
            return {'anomaly_level_hotelling': hotelling,
                    'control_limit_hotelling': self._hotelling_limit_p2,
                    'anomaly_level_spe': SPE,
                    'control_limit_spe': self._spe_limit,
                    }

        # Merge the contributions of the variables to the T2 and SPE statistics
        t2_contributions = t2_contributions.set_index('variable')
        spe_contributions = spe_contributions.set_index('variable')

        contributions_df = t2_contributions.join(spe_contributions, lsuffix='_t2', rsuffix='_spe')
            
        return {'anomaly_level_hotelling': hotelling,
                 'control_limit_hotelling': self._hotelling_limit_p2,
                 'anomaly_level_spe': SPE,
                'control_limit_spe': self._spe_limit,
                'contributions': contributions_df.to_dict(orient='index')}
    
    @validate_dataframe('observation')
    def t2_contribution(self, observation: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the contributions of each variable to the T2 statistic for a given observation.
    
        Parameters
        ----------
        observation : pd.DataFrame
            The observation to analyze. Must be a single observation (1 row) in a pandas DataFrame format.
    
        Returns
        -------
        pd.DataFrame
            DataFrame containing the contributions of each variable to the T2 statistic.
        """
        hotelling = self.hotelling_t2(observation)
        projected_scores = self.transform(observation)
        normalized_scores = self._calculate_normalized_scores(projected_scores)
        high_scores = self._get_high_scores(normalized_scores)
        contributions_df = self._calculate_contributions(observation, projected_scores, high_scores)
        return contributions_df.sort_values("contribution", ascending=False), hotelling
    
    def _calculate_normalized_scores(self, projected_scores: pd.DataFrame) -> np.ndarray:
        """
        Calculates the normalized scores for the projected data.
    
        Parameters
        ----------
        projected_scores : pd.DataFrame
            The projected scores of the observation.
    
        Returns
        -------
        np.ndarray
            The normalized scores.
        """
        normalized_scores = projected_scores**2 / self._eigenvals
        normalized_scores /= np.max(normalized_scores)
        return normalized_scores
    
    def _get_high_scores(self, normalized_scores: np.ndarray) -> np.ndarray:
        """
        Identifies the high scores from the normalized scores.
    
        Parameters
        ----------
        normalized_scores : np.ndarray
            The normalized scores.
    
        Returns
        -------
        np.ndarray
            The indices of the high scores.
        """
        return np.where(normalized_scores > 0)[1]
    
    def _calculate_contributions(self, observation: pd.DataFrame, projected_scores: pd.DataFrame, high_scores: np.ndarray) -> pd.DataFrame:
        """
        Calculates the contributions of each variable to the T2 statistic.
    
        Parameters
        ----------
        observation : pd.DataFrame
            The observation to analyze.
        projected_scores : pd.DataFrame
            The projected scores of the observation.
        high_scores : np.ndarray
            The indices of the high scores.
    
        Returns
        -------
        pd.DataFrame
            DataFrame containing the contributions of each variable to the T2 statistic.
        """
        truncated_loadings = self._loadings.values[:, high_scores].T
        truncated_scores = projected_scores.values[:, high_scores]
        truncated_eigenvals = self._eigenvals[high_scores]
        mean_diff = (observation - self._mean_train).values

        pc_features = ['PC_' + str(i+1) for i in high_scores]

        loadings_mean = (truncated_loadings * mean_diff)

        contributions = np.zeros(loadings_mean.shape)
        for i in range(loadings_mean.shape[0]):
            contributions[i, :] = loadings_mean[i, :] * (truncated_scores[0, i] / truncated_eigenvals[i])

        contributions_df = pd.DataFrame(contributions, columns=self._variables, index=pc_features)

        #If the contribution is negative, set it to 0
        contributions_df[contributions_df < 0] = 0

        return pd.DataFrame({'variable': self._variables, 'contribution': contributions_df.mean(axis=0)})
    
    def spe_contribution(self, observation: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the contributions of each variable to the SPE (Squared Prediction Error) statistic for a given observation.
    
        Parameters
        ----------
        observation : pd.DataFrame
            The observation to analyze. Must be a single observation (1 row) in a pandas DataFrame format.
    
        Returns
        -------
        pd.DataFrame
            DataFrame containing the contributions of each variable to the SPE statistic.
        """
        SPE, residuals = self.spe(observation)
        contributions_df = self._calculate_spe_contributions(residuals)
        return contributions_df, SPE
    
    def dmodx_contribution(self, observation: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the contributions of each variable to the DmodX statistic for a given observation.
    
        This method computes the contribution of each variable to the DmodX (Distance to Model in X-space) by:
          1. Transforming the observation via PCA and reconstructing it.
          2. Calculating the residuals between the actual observation and its reconstruction.
          3. Computing DmodX for the observation.
          4. Determining the contributions of each variable from the residuals using an internal helper function.
    
        Parameters
        ----------
        observation : pd.DataFrame
            A pandas DataFrame representing a single observation. The DataFrame should have the same 
            features as those used to fit the PCA model.
    
        Returns
        -------
        tuple[pd.DataFrame, float]
            A tuple where the first element is a DataFrame containing the contributions of each variable 
            to the DmodX statistic, and the second element is the computed DmodX value for the observation.
        """
        reconstructed = self.inverse_transform(self.transform(observation))
        residuals = observation - reconstructed
    
        dmodx = self.dmodx(observation)
    
        contributions_df = self._calculate_dmodx_contributions(residuals)
    
        return contributions_df, dmodx

    
    def _calculate_spe_contributions(self, residuals: np.ndarray) -> pd.DataFrame:
        """
        Calculate the contributions of each variable to the Squared Prediction Error (SPE) statistic.
        
        For each variable, its contribution is computed as the square of its residual value.
        If residuals are provided for multiple observations (2D array), the contributions are summed
        across all samples. The output DataFrame provides both the absolute and relative contributions
        of each variable.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from the original minus the reconstructed data. This can be a 1D array 
            (for a single observation) or a 2D array of shape (n_samples, n_features) where contributions 
            are aggregated across samples.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                - variable: Name of the variable.
                - contribution: The absolute contribution to the SPE.
                - relative_contribution: Fraction of the total SPE contributed by the variable.
        """
        # Compute squared contributions.
        # If residuals is 2D, sum across samples for each variable.
        if residuals.ndim == 1:
            squared_contributions = residuals ** 2
        else:
            squared_contributions = np.sum(residuals ** 2, axis=0)
    
        # Create a DataFrame for the contributions.
        contributions_df = pd.DataFrame({
            'variable': self._variables,
            'contribution': squared_contributions
        })
    
        # Compute relative contributions.
        total_contribution = contributions_df['contribution'].sum()
        contributions_df['relative_contribution'] = contributions_df['contribution'] / total_contribution
    
        # Sort variables by absolute contribution in descending order.
        contributions_df.sort_values('contribution', ascending=False, inplace=True)
        contributions_df.reset_index(drop=True, inplace=True)
    
        return contributions_df
    
    def _calculate_dmodx_contributions(self, residuals: np.ndarray) -> pd.DataFrame:
        """
        Calculate the contributions of each variable to the DmodX (Distance to Model in X-space) statistic.
    
        This method computes the contribution of each variable to the overall model deviation by:
          1. Computing the sum of squares of the residuals from the training fit.
          2. Calculating a weight (w_k) as the square root of the explained sum of squares.
          3. Multiplying the weight with the residuals of the observation.
          4. Forming a DataFrame that contains each variable's absolute and relative contribution.
          5. Sorting the contributions in descending order by absolute contribution.
    
        Parameters
        ----------
        residuals : np.ndarray
            An array of residuals for the observation(s). Typically, this is obtained by subtracting the 
            reconstructed observation from the original observation.
    
        Returns
        -------
        pd.DataFrame
            A DataFrame with columns:
                - 'variable': The name of each variable.
                - 'contribution': The absolute contribution of the variable to the DmodX.
                - 'relative_contribution': The fraction of the total DmodX contributed by the variable.
        """
        # Calculate the sum of squares of the residuals from the training fit.
        explained_sum_squares = np.sum((self._residuals_fit.values) ** 2)
        # Compute weight as the square root of the explained sum of squares.
        w_k = np.sqrt(explained_sum_squares)
    
        # Multiply the weight with the residuals; assume residuals is a 2D array and take the first row.
        contribution = w_k * residuals.values
    
        # Create a DataFrame for absolute contributions.
        contributions_df = pd.DataFrame({
            'variable': self._variables,
            'contribution': contribution[0]
        })
    
        # Compute relative contributions.
        total_contribution = contributions_df['contribution'].sum()
        contributions_df['relative_contribution'] = contributions_df['contribution'] / total_contribution
    
        # Sort variables by absolute contribution in descending order.
        contributions_df.sort_values('contribution', ascending=False, inplace=True)
        contributions_df.reset_index(drop=True, inplace=True)
    
        return contributions_df
    
    def generate_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data samples based on the PCA model.
    
        This method generates synthetic observations by sampling from the PCA subspace using a 
        multivariate normal distribution. The means and standard deviations of the latent scores 
        (obtained from the fitted PCA model) define the distribution. Under the assumption of independence 
        (i.e. diagonal covariance), samples are drawn in the latent space and then transformed back to the 
        original feature space using the inverse PCA transformation.
    
        Parameters
        ----------
        num_samples : int, optional
            Number of synthetic samples to generate (default is 1000).
    
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the generated synthetic data in the original feature space.
        """
        # Define latent variable names (e.g., PC_1, PC_2, ..., PC_n)
        pc_features = [f'PC_{i}' for i in range(1, self._ncomps + 1)]
        
        # Compute the mean and standard deviation of the PCA scores.
        means = self._scores.mean()
        stds = self._scores.std()
    
        # Construct a diagonal covariance matrix (i.e., assuming independent components).
        covariance = np.diag(stds**2)
    
        # Sample from the multivariate normal in the PCA subspace.
        latent_samples = np.random.multivariate_normal(means, covariance, size=num_samples)
        latent_df = pd.DataFrame(latent_samples, columns=pc_features)
    
        # Transform the latent samples back into the original feature space.
        return self.inverse_transform(latent_df)
    
    def dmodx(self, data: pd.DataFrame, normalize: bool = True) -> np.ndarray:
        """
        Calculate the DModX statistic for each observation in the dataset.
    
        DModX (Distance to Model in X-space) quantifies the distance of an observation from 
        the PCA model's fitted subspace (i.e., the reconstruction error), thereby indicating 
        its leverage or potential influence on the model. Higher DModX values suggest that 
        an observation deviates more significantly from the PCA model and may be considered an outlier.
        
        The computation is based on the Squared Prediction Error (SPE) scaled by the degrees 
        of freedom available (number of variables minus number of principal components). Optionally, 
        the DModX values can be normalized by an estimate of the residual standard deviation.
    
        Parameters
        ----------
        data : pd.DataFrame
            The input data for which the DModX statistic will be calculated. Each row should represent 
            one observation with the same variables used to fit the PCA model.
        normalize : bool, optional
            If True, the computed DModX values are normalized by the estimated residual standard deviation.
            The default is True.
    
        Returns
        -------
        np.ndarray
            A one-dimensional NumPy array containing the DModX values for each observation in the input data.
    
        Notes
        -----
        Let:
            m = number of observations,
            A = number of principal components,
            K = number of variables.
        Then:
            DModX = sqrt( SPE / (K - A) )
        where SPE is the Squared Prediction Error for each observation.
        If normalization is enabled, DModX is further divided by:
            s₀ = sqrt( Sum_{i,j}(residuals^2) / ((m - A - 1) * (K - A)) )
        """
        
        # Calculate the Squared Prediction Error (SPE) and obtain the residuals.
        SPE, residuals = self.spe(data)
    
        m = self._nobs
        A = self._ncomps
        K = self._nvars
    
        dmodx_values = np.sqrt(np.array(SPE) / (K - A))
    
        if normalize:
            s_0 = np.sqrt(np.sum(residuals.values**2) / ((m - A - 1) * (K - A)))
            dmodx_values = dmodx_values / s_0
    
        return dmodx_values

    '''
    PLOTS
    '''
    @require_fitted
    def score_plot(self, comp1:int, comp2:int, hue:pd.Series=None, test_set:pd.DataFrame=None):
        '''
        Generates a score plot of the selected components

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
        if comp1 <= 0 or comp1>self._nvars or comp2 <= 0 or comp2>self._nvars:
            raise NComponentsError(self._nvars)
        
        if test_set is not None:

            test_set = self.transform(test_set)

        return score_plot(scores=self._scores, comp1=comp1, comp2=comp2, explained_variance=self._explained_variance, hue=hue, index_name=self._index_name, test_set=test_set)
    
    @require_fitted
    def biplot(self, comp1:int, comp2:int, hue:pd.Series=None, test_set:pd.DataFrame=None):
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
        if comp1 <= 0 or comp1>self._ncomps or comp2 <= 0 or comp2>self._ncomps:
            raise NComponentsError(self._nvars)
        
        if test_set is not None:

            test_set = self.transform(test_set)
    
        return biplot(scores=self._scores, loadings=self._loadings, comp1=comp1, comp2=comp2, explained_variance=self._explained_variance, hue=hue, index_name=self._index_name, test_set=test_set)
    
    @require_fitted
    def loadings_barplot(self, comp:int):
        '''
        Generates a bar plot of the loadings of the selected component

        Parameters
        ----------
        comp : int
            The number of the component.

        Returns
        -------
        None
        '''
        if comp <= 0 or comp>self._ncomps:
            raise NComponentsError(self._nvars)

        # Altair plot for the loadings
        return loadings_barplot(self._loadings, self._explained_variance, comp)

    @require_fitted
    @validate_dataframe('X_obs')
    def difference_plot(self, 
                        X_obs:pd.DataFrame):   
        '''
        Generates a bar plot visualizing the difference between a specific observation and the mean of the sample.

        This method creates a bar plot to visually compare the values of a given observation against the mean values of the training sample. It is particularly useful for understanding how an individual observation deviates from the average trend across each variable. The method checks if the input observation is a pandas DataFrame and if it contains the same variables as those used in the training data. Optionally, if the model is set to standardize data, the observation will be preprocessed accordingly before plotting.

        Parameters
        ----------
        X_obs : pd.DataFrame
            The observation to compare against the sample mean. Must be a single observation (1 row) in a pandas DataFrame format.

        Returns
        -------
        alt.Chart
            An Altair Chart object that represents the bar plot. This object can be directly displayed in Jupyter notebooks or saved as an image.

        Raises
        ------
        NotDataFrameError
            If `X_obs` is not a pandas DataFrame.
        ValueError
            If `X_obs` does not contain the same variables as the training data.
        '''       
        if sorted(X_obs.columns) != sorted(self._variables):
            raise ValueError("The observation must have the same variables as the training data")
            
        X_transform = X_obs.copy()
        if self._standardize:
            X_transform = self.preprocess(data=X_transform)

        df_plot = pd.DataFrame({'variable': self._variables, 'value': X_transform.values[0]})
        # Altair plot for the differences
        return alt.Chart(df_plot).mark_bar().encode(
            x=alt.X('variable', title='Variable'),
            y=alt.Y('value', title='Difference with respect to the mean (std)'),
            tooltip=['variable', 'value']
        ).interactive()

    @require_fitted
    def hotelling_t2_plot_p1(self):
        '''
        Generates an interactive plot visualizing the Hotelling's T2 statistic over observations.

        This method creates an interactive line plot of the Hotelling's T2 statistic for each observation in the dataset. The plot includes a horizontal dashed line indicating the threshold value beyond which an observation is considered an outlier. This visualization aids in identifying observations that significantly deviate from the model's assumptions or the majority of the data.

        The plot is generated using Altair, a declarative statistical visualization library for Python. The method configures the plot with a title that includes the significance level (alpha) and the threshold value for the Hotelling's T2 statistic. Observations and their corresponding T2 values are displayed as tooltips when hovering over the plot.

        Returns
        -------
        alt.LayerChart
            An Altair LayerChart object that combines the line plot of Hotelling's T2 statistics with the threshold rule. This object can be displayed in Jupyter notebooks or saved as an image.

        Notes
        -----
        - The method assumes that the Hotelling's T2 statistics (`self._hotelling`) and the threshold (`self._hotelling_limit_p1`) have been previously calculated and are stored as attributes of the class.
        - The plot is interactive, allowing for zooming and panning to explore the data points in detail.
        '''      
        hotelling_df = pd.DataFrame({'observation': range(self._nobs), 'T2': self._hotelling})

        return generic_stat_plot(data=hotelling_df, 
                                 stat_column="T2",
                                 control_limit=self._hotelling_limit_p1,
                                 phase="Phase I",
                                 alpha=self._alpha)
    
    @require_fitted
    @validate_dataframe('test_set')
    def hotelling_t2_plot_p2(self, test_set:pd.DataFrame):
        '''
        Generates an interactive plot of the Hotelling's T2 statistic for Phase II observations.

        This method visualizes the Hotelling's T2 statistic for a given test set, aiding in the identification of outliers. It checks if the test set is a pandas DataFrame and raises an error if not. The method projects the test set onto the PCA model, calculates the Hotelling's T2 statistic for each observation, and generates an interactive line plot with a threshold line indicating the outlier cutoff.

        The plot is created using Altair, enabling interactive exploration through zooming and panning. Observations are plotted along the x-axis, with their corresponding Hotelling's T2 values on the y-axis. A red dashed line represents the threshold value, above which observations are considered outliers. Tooltips display the observation index and T2 value upon hovering.

        Parameters
        ----------
        test_set : pd.DataFrame
            The test set to be analyzed. Must be a pandas DataFrame.

        Returns
        -------
        alt.LayerChart
            An Altair LayerChart object combining the Hotelling's T2 statistic line plot with the threshold rule. This object can be displayed in Jupyter notebooks or saved as an image.

        Raises
        ------
        NotDataFrameError
            If `test_set` is not a pandas DataFrame.

        Notes
        -----
        - The method assumes the PCA model has been fitted and the threshold (`self._hotelling_limit_p2`) has been set.
        - The plot's title includes the significance level (alpha) and the threshold value, providing context for the analysis.
        '''
        hotelling = self.hotelling_t2(test_set)
        hotelling_df = pd.DataFrame({'observation': range(self._nobs), 'T2': hotelling})

        return generic_stat_plot(data=hotelling_df, 
                                 stat_column="T2",
                                 control_limit=self._hotelling_limit_p2,
                                 phase="Phase II",
                                 alpha=self._alpha)
    
    @require_fitted
    def spe_plot_p1(self):
        '''
        Generates an interactive plot visualizing the Squared Prediction Error (SPE) statistic for Phase I observations.

        This method plots the SPE statistic for each observation in the dataset, providing a visual tool for outlier detection. The plot includes a horizontal dashed line indicating the SPE threshold, which helps identify observations with unusually high prediction errors, suggesting they may be outliers.

        The plot is generated using Altair, a declarative visualization library, ensuring interactivity with features like zooming and panning. Observations are plotted along the x-axis, with their SPE values on the y-axis. A red dashed line represents the SPE threshold, above which observations are considered potential outliers. Tooltips display the observation index and SPE value for further inspection.

        Parameters
        ----------
        None

        Returns
        -------
        alt.LayerChart
            An Altair LayerChart object that combines the SPE statistic line plot with the threshold rule. This object can be displayed in Jupyter notebooks or saved as an image.

        Notes
        -----
        - The method assumes that the SPE statistics (`self._spe`) and the threshold (`self._spe_limit`) have been previously calculated and are stored as attributes of the class.
        - The plot's title includes the significance level (alpha) and the threshold value, providing context for the analysis.
        '''
        spe_df = pd.DataFrame({'observation': range(self._nobs), 'SPE': self._spe})

        return generic_stat_plot(data=spe_df, 
                                 stat_column="SPE",
                                 control_limit=self._spe_limit,
                                 phase="Phase I",
                                 alpha=self._alpha)
    
    @require_fitted
    @validate_dataframe('test_set')
    def spe_plot_p2(self, test_set:pd.DataFrame):
        '''
        Generates an interactive plot visualizing the Squared Prediction Error (SPE) statistic for Phase II observations.

        This method creates an interactive plot of the SPE statistic for each observation in the provided test set. It serves as a diagnostic tool to identify observations with unusually high prediction errors, which may indicate outliers or anomalies. The plot includes a horizontal dashed line indicating the SPE threshold, aiding in the visual identification of potential outliers.

        The plot is generated using Altair, a declarative visualization library, ensuring interactivity with features like zooming and panning. Observations are plotted along the x-axis, with their SPE values on the y-axis. A red dashed line represents the SPE threshold, above which observations are considered potential outliers. Tooltips provide detailed information about the observation index and SPE value for further inspection.

        Parameters
        ----------
        test_set : pd.DataFrame
            The test set to be analyzed. Must be a pandas DataFrame.

        Returns
        -------
        alt.LayerChart
            An Altair LayerChart object that combines the SPE statistic line plot with the threshold rule. This object can be displayed in Jupyter notebooks or saved as an image.

        Raises
        ------
        NotDataFrameError
            If `test_set` is not a pandas DataFrame.

        Notes
        -----
        - The method assumes that the SPE statistics and the threshold (`self._spe_limit`) have been previously calculated and are stored as attributes of the class.
        - The plot's title includes the significance level (alpha) and the threshold value, providing context for the analysis.
        '''
        phase = 'Phase II'
        SPE, _ = self.spe(test_set)

        nobs = len(SPE)
        spe_df = pd.DataFrame({'observation': range(nobs), 'SPE': SPE})

        return generic_stat_plot(data=spe_df, 
                                 stat_column="SPE",
                                 control_limit=self._spe_limit,
                                 phase="Phase II",
                                 alpha=self._alpha)
    
    @require_fitted
    def dmodx_plot_p1(self) -> alt.Chart:
        """
        Generate an interactive plot for DModX (Distance to Model in X-space) statistics for Phase I.
    
        Phase I corresponds to the training phase, where the PCA model's performance is assessed via DModX.
        This method creates an interactive Altair plot that visualizes the DModX values for each training observation,
        along with a control limit line. The control limit is computed based on the pre-configured significance level (alpha)
        and serves as a threshold for identifying potential model deviations or outliers.
    
        Returns
        -------
        alt.Chart
            An interactive Altair chart displaying DModX values for each observation and the DModX control limit line.
        """      
        # Create a DataFrame containing observation indices and their corresponding DModX values.
        dmodx_df = pd.DataFrame({'observation': range(self._nobs), 'DModX': self._dmodx})
        
        # Create and return the interactive Altair plot using the dedicated plotting function.
        return generic_stat_plot(data=dmodx_df, 
                                 stat_column="DModX",
                                 control_limit=self._dmodx_limit,
                                 phase="Phase I",
                                 alpha=self._alpha)
    
    @validate_dataframe('test_set')
    @require_fitted
    def dmodx_plot_p2(self, test_set: pd.DataFrame) -> alt.Chart:
        """
        Generate an interactive Altair plot for DModX (Distance to Model in X-space) statistics for Phase II (test set).
    
        Phase II corresponds to the evaluation phase where the PCA model is applied to new, unseen data.
        This method calculates the DModX values for each observation in the test set and visualizes them along with
        a control limit line. The control limit, determined during model training, acts as a threshold to flag potential 
        outliers or observations that deviate significantly from the model.
    
        Parameters
        ----------
        test_set : pd.DataFrame
            A DataFrame containing the test data. Each row should represent one observation with the same features 
            as those used to fit the PCA model.
    
        Returns
        -------
        alt.Chart
            An interactive Altair chart displaying DModX values for each test observation and the corresponding DModX 
            control limit line.
        """
        # Compute DModX for the test set.
        dmodx = self.dmodx(test_set)
        
        # Use the length of the computed dmodx array to create observation indices.
        dmodx_df = pd.DataFrame({'observation': range(len(dmodx)), 'DModX': dmodx})
        
        # Generate and return the Altair plot using the dedicated plotting function.
        return generic_stat_plot(data=dmodx_df, 
                                 stat_column="DModX",
                                 control_limit=self._dmodx_limit,
                                 phase="Phase II",
                                 alpha=self._alpha)


    @validate_dataframe('data')
    @require_fitted
    def residual_barplot(self, data:pd.DataFrame):
        '''
        Generates an interactive bar plot visualizing the residuals for a specific observation within the dataset.

        This method creates a bar plot to display the residuals (differences between observed and predicted values) for a single observation in the dataset. It is designed to help in diagnosing and understanding the prediction errors for individual observations. The method first checks if the input data is a pandas DataFrame and if it contains the correct number of features as expected by the model. It also verifies that exactly one observation is provided for analysis.

        Upon validation, the method calculates the residuals and the Squared Prediction Error (SPE) for the given observation. It then generates an interactive bar plot using Altair, where each bar represents a variable's residual. The plot includes tooltips for detailed residual values and is titled with the observation's index and its SPE value, providing immediate insight into the model's performance on that observation.

        Parameters
        ----------
        data : pd.DataFrame
            The observation to analyze, provided as a single-row pandas DataFrame.

        Returns
        -------
        alt.Chart
            An interactive Altair Chart object representing the residuals bar plot. This plot includes tooltips and is titled with the observation's index and SPE value.

        Raises
        ------
        NotDataFrameError
            If the input `data` is not a pandas DataFrame.
        ValueError
            If the number of features in `data` does not match the model's expected number of features.
        ValueError
            If `data` contains more than one observation.

        Notes
        -----
        - The method assumes that the model and its variables (`self._variables`) have been previously defined.
        - The SPE (Squared Prediction Error) is calculated as part of the residuals analysis and is displayed in the plot title for reference.
        '''   
        if data.shape[1] != len(self._variables):
            raise ValueError(f'Number of features in data must be {len(self._variables)}')
        
        if data.shape[0] != 1:
            raise ValueError(f'Number of observations in data must be 1')
        
        SPE, residuals = self.spe(data)

        residuals = residuals.T.reset_index()
        residuals.columns = ['variable', 'residual']
        obs_name = data.index.values[0]
    
        # Altair plot for the residuals
        return residuals_barplot(residuals, SPE, obs_name)
    
    @require_fitted
    @validate_dataframe('observation')
    def spe_contribution_plot(self, observation: pd.DataFrame):
        """
        Generates an interactive bar plot visualizing each variable's contribution to the Squared Prediction Error (SPE) for a specific observation.
    
        This method creates a bar plot that breaks down the contribution of each variable to the overall SPE of a given observation. It is useful for identifying which variables contribute most to the observation's deviation from the model's predictions.
    
        Parameters
        ----------
        observation : pd.DataFrame
            The observation to analyze. Must be a single observation (1 row) in a pandas DataFrame format.
    
        Returns
        -------
        alt.Chart
            An Altair Chart object representing the interactive bar plot of variable contributions to the SPE.
        pd.DataFrame
            The DataFrame containing the contributions of each variable to the SPE.
    
        Raises
        ------
        ValueError
            If the number of features in `observation` does not match the number of variables in the model.
            If `observation` contains more than one observation.
    
        Notes
        -----
        - The plot includes tooltips for each variable's contribution and displays the total SPE value in the title.
        """
        if observation.shape[1] != len(self._variables):
            raise ValueError(f'Number of features in data must be {len(self._variables)}')
    
        if observation.shape[0] != 1:
            raise ValueError(f'Number of observations in data must be 1')
    
        contributions_df, SPE = self.spe_contribution(observation)
    
        obs_name = observation.index.values[0]
    
        return spe_contribution_plot(contributions_df, SPE, obs_name)
    
    def hotelling_t2_contribution_plot(self, observation: pd.DataFrame) -> alt.Chart:
        """
        Generate an interactive Altair plot showing the contributions of each variable to Hotelling's T² statistic.
    
        The function validates that the input contains exactly one observation and that the number of features 
        matches the variables from the fitted model. It then computes the contributions using the t2_contribution 
        method and passes the results to the plotting function for visualization.
    
        Parameters
        ----------
        observation : pd.DataFrame
            A DataFrame representing a single observation. It must have exactly one row and the same number of 
            columns as the fitted PCA model (i.e., len(self._variables)).
    
        Returns
        -------
        alt.Chart
            An interactive Altair chart visualizing the contributions of each variable to Hotelling's T² statistic 
            for the given observation.
    
        Raises
        ------
        ValueError
            If the number of features in the observation does not match the number of model variables or if the 
            DataFrame does not contain exactly one observation.
        """
        # Validate that the observation has the expected number of features.
        if observation.shape[1] != len(self._variables):
            raise ValueError(f"Input data must have {len(self._variables)} features, but got {observation.shape[1]}.")
    
        # Validate that there is exactly one observation.
        if observation.shape[0] != 1:
            raise ValueError("Input data must contain exactly one observation.")
    
        # Calculate contributions and Hotelling's T² statistic.
        contributions_df, hotelling = self.t2_contribution(observation)
        
        # Extract observation name (assumes the index is informative).
        obs_name = observation.index.values[0]
    
        # Generate and return the Altair plot for Hotelling's T² contributions.
        return hotelling_t2_contribution_plot(contributions_df, hotelling, obs_name)
    
    def dmodx_contribution_plot(self, observation: pd.DataFrame) -> alt.Chart:

        contributions_df, dmodx = self.dmodx_contribution(observation)

        obs_name = observation.index.values[0]

        return dmodx_contribution_plot(contributions_df, dmodx, obs_name)
    
    def actual_vs_predicted(self, observation: pd.DataFrame) -> alt.Chart:
        """
        Generate an interactive Altair chart comparing Actual versus Predicted values for a given observation.
    
        This method first transforms the input observation into the PCA latent space and then reconstructs it back to the original space, producing predicted values. It compares these predictions with the actual values of the observation for each variable. The resulting data is reshaped into a long format, and an interactive grouped bar chart is generated using an external Altair plotting function.
    
        Parameters
        ----------
        observation : pd.DataFrame
            A DataFrame representing a single observation. It must have exactly one row and the same number of 
            features as the fitted PCA model (i.e., len(self._variables)).
    
        Returns
        -------
        alt.Chart
            An interactive Altair chart displaying side-by-side bars comparing Actual and Predicted values for each variable.
    
        Raises
        ------
        ValueError
            If the observation does not contain exactly one row or if the number of features does not match the model.
        """
        # Validate input dimensions
        if observation.shape[0] != 1:
            raise ValueError("Input observation must contain exactly one row.")
        if observation.shape[1] != len(self._variables):
            raise ValueError(f"Input observation must have {len(self._variables)} features, but got {observation.shape[1]}.")
    
        # Generate predicted values by applying the PCA transformation and its inverse.
        predicted_values = self.inverse_transform(self.transform(observation))
    
        # Transpose the predicted values DataFrame so that each variable becomes a row.
        predicted_df = predicted_values.T.copy()
        predicted_df.columns = ['Predicted']
    
        # Add Actual values from the observation (transposed to align with variables).
        predicted_df['Actual'] = observation.T.values.flatten()
    
        # Reset index to convert the variable names (index) into a column.
        predicted_df = predicted_df.reset_index().rename(columns={'index': 'Variable'})
    
        # Melt the DataFrame to long format for Altair; now each row represents a variable-type pair.
        melted_df = pd.melt(predicted_df,
                            id_vars='Variable',
                            value_vars=['Predicted', 'Actual'],
                            var_name='Type',
                            value_name='Value')
    
        # Generate and return the interactive Altair chart using the external plotting function.
        return actual_vs_predicted(melted_df)
