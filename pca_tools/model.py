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
import pdb
from sklearn.decomposition import PCA as PCA_sk

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_comps:int=None, 
                 numerical_features:list=[], 
                 standardize:bool=True, 
                 tolerance:float=1e-4, 
                 verbose:bool=False,
                 alpha:float=.99) -> None:
        
        if not 0 < tolerance < 1:
            raise ValueError('Tolerance must be strictly between 0 and 1')
        
        if not 0 < alpha < 1:
            raise ValueError('Alpha must be strictly between 0 and 1')

        if not isinstance(numerical_features, list):
            raise NotAListError(type(numerical_features).__name__)
        
        if not isinstance(verbose, bool) or not isinstance(standardize, bool):
            raise NotBoolError()  

        
        self._standardize = standardize
        self._tolerance = tolerance
        self.verbose = verbose
        self._ncomps = n_comps
        self._numerical_features = numerical_features
        self._alpha = alpha
        self.model = PCA_sk(n_components=self._ncomps, svd_solver='full', tol=self._tolerance, iterated_power='auto')

        self._scaler = Pipeline([
            # ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        return params
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_rsquared_acc(self):
        return self._rsquared_acc

    def fit(self, data, y=None):

        if not isinstance(data, pd.DataFrame):
            raise NotDataFrameError(type(data).__name__)

        if self._standardize:
            if not self._numerical_features:
                self._numerical_features = data.columns.tolist()
            
            self._scaler.fit(data[self._numerical_features])

        self.train(data)

        self._spe, _ = self.spe(data)
        self._hotelling = self.hotelling_t2(data)

        self.control_limits(alpha=self._alpha)

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

        index = data.index
        columns = data.columns

        X_transform = data.copy()

        if self._numerical_features:
            X_transform[self._numerical_features] = self._scaler.transform(X_transform[self._numerical_features])
        else:
            X_transform = pd.DataFrame(self._scaler.transform(X_transform), columns=columns, index=index)

        return X_transform
            
    def train(self, data:pd.DataFrame):
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
        if self._ncomps is None:
            self._ncomps = data.shape[1]

        if self._ncomps <= 0 or self._ncomps > data.shape[1]:
            raise NComponentsError(data.shape[1])

        if not isinstance(data, pd.DataFrame):
            raise NotDataFrameError(type(data).__name__)

        self._variables = data.columns
        self._index = data.index
        self._index_name = data.index.name
        self._nobs, self._nvars = data.shape

        X = data.copy()
        if self._standardize:
            X = self.preprocess(data=X)

        X = X.values

        self.model.fit(X)

        self._loadings = pd.DataFrame(self.model.components_.T, columns=[f"PC_{i+1}" for i in range(self._ncomps)], index=self._variables)
        self._scores = pd.DataFrame(self.model.transform(X), columns=[f"PC_{i+1}" for i in range(self._ncomps)], index=self._index)
        self._explained_variance = self.model.explained_variance_ratio_
        self._rsquared_acc = np.cumsum(self.model.explained_variance_ratio_)
        self._eigenvals = np.var(self._scores.values, axis=0)
        self._residuals_fit = X - self._scores @ self._loadings.T
        self._mean_train = np.mean(data.values, axis=0)
        self._std_train = np.std(data.values, axis=0)

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

        if not isinstance(data, pd.DataFrame):
            raise NotDataFrameError(type(data).__name__)

        X_transform = data.copy()
        # Descale and demean matrix
        if self._standardize:
            X_transform = self.preprocess(data=X_transform)
        
        return pd.DataFrame(X_transform @ self._loadings, columns=self._scores.columns, index=data.index)

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

    def inverse_transform(self, data):
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

        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()

        if self._standardize==True:
            if self._numerical_features:
                result = data @ self._loadings.T
                result = self._scaler.inverse_transform(result[self._numerical_features])
            else:
                result = self._scaler.inverse_transform(data @ self._loadings.T)
        else:
            result = data @ self._loadings.T

        return pd.DataFrame(result, columns=self._variables, index=data.index)
    
    def control_limits(self, alpha:float=0.95):
        '''
        Calculates the control limits for Hotelling's T2 and SPE (Squared Prediction Error) statistics.

        This method computes the control limits for both the Hotelling's T2 statistic and the SPE statistic for a PCA model. The control limits are calculated for two phases: Phase I (initial control limit based on the model fitting dataset) and Phase II (control limit for monitoring new observations). These limits are used to detect outliers or shifts in the process mean or variability in multivariate data.

        Parameters
        ----------
        alpha : float, optional
            The significance level used to calculate the control limits. It represents the probability of observing a point within the control limits under the assumption that the process is in control. The default value is 0.95, corresponding to a 95% confidence level.

        Attributes Set
        --------------
        _hotelling_limit_p1 : float
            The control limit for Hotelling's T2 statistic in Phase I.
        _hotelling_limit_p2 : float
            The control limit for Hotelling's T2 statistic in Phase II.
        _spe_limit : float
            The control limit for the SPE statistic.

        Notes
        -----
        - The Hotelling's T2 statistic is a multivariate analogue of the univariate t-square statistic. It is used to test the hypothesis that the mean vector of a multivariate population is equal to a given vector.
        - The SPE statistic measures the squared prediction error of each observation from the PCA model. It is used to detect observations that do not conform to the model.
        - The control limits are based on the F-distribution for Hotelling's T2 and the chi-squared distribution for SPE, adjusted for the sample size and the number of principal components in the model.
        '''
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        # Hotelling's T2 control limit. Phase I
        dfn = self._ncomps/2
        dfd = (self._nobs-self._ncomps-1)/2
        const = ((self._nobs-1)**2)/self._nobs

        self._hotelling_limit_p1 = beta.ppf(alpha, dfn, dfd)*const

        # Hotelling's T2 control limit. Phase II
        const = (self._ncomps * (self._nobs**2 -1)) / (self._nobs * (self._nobs - self._ncomps))
        dfn = self._ncomps
        dfd = self._nobs - self._ncomps

        self._hotelling_limit_p2 = f.ppf(alpha, dfn, dfd)*const

        # SPE control limit
        b, nu = np.mean(self._spe), np.var(self._spe)
        
        df = (2*b**2)/nu
        const = nu/(2*b)

        self._spe_limit = chi2.ppf(alpha, df)*const
    
    def hotelling_t2(self, data):
        '''
        Hotelling's T2 represents the estimated squared Mahalanobis distance from the center of the latent subspace
        to the projection of an observation onto this subspace (Ferrer, 2014).
        We calculate this statistic as T2 = sum(t_a**2/lambda_a), being lambda_a the eigenvalue of the ath component
        and t_a the score of the ith observation in the ath component.
        Under the assumption that the scores follow a multivariate normal distribution, it holds (Tracy, 1992), that in Phase I,
        T2 follows a beta distribution with A/2 and (m-A-1)/2 degrees of freedom, being A the number of components
        and m the number of observations

        This statistic help us to detect outliers given that it is a measure of the distance between the centroid
        of the subspace and an observation. Observations that are above the control limit might be considered extreme outliers

        Returns
        -------
        Hotelling's T2 statistic for every observation.
        '''
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
        if not isinstance(data, pd.DataFrame):
            raise NotDataFrameError(type(data).__name__)

        predicted_scores = self.transform(data)

        return list(np.sum((predicted_scores.values**2) / self._eigenvals, axis=1))
        
    def spe(self, data):
        '''
        Represents the sum of squared prediction errors. The value is given by the expression 
        e^T_i * e_i, so the SPE statistic is the scalar product of the residuals vector of observation i
        multiplied by its transposed self.
        
        The SPE statistic represents the squared Euclidean distance of an observation from the generated
        subspace, and gives a measure of how close the observation is to the A-dimensional subspace.
        
        Observations that are above the control limit can be considered as moderate outliers.

        Returns
        -------
        SPE statistic for every observation.

        '''

        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
        X_transform = data.copy()

        # Descale and demean matrix
        if self._standardize:
            X_transform = self.preprocess(data=X_transform)

        X_transform = X_transform.values

        predicted_scores = self.transform(data)

        # SPE statistic

        residuals = X_transform - predicted_scores.values @ self._loadings.values.T
        SPE = np.sum(np.square(residuals), axis=1)

        return SPE.tolist(), residuals

    def project(self, X_predict):
        """
        Projects new data onto the fitted PCA model and calculates Hotelling's T2 and SPE statistics.

        This method is used to project new observations onto the principal components obtained from the fitted PCA model. 
        It also calculates the Hotelling's T2 statistic and the Squared Prediction Error (SPE) for each observation, 
        which can be used for anomaly detection or assessing the fit of the model.

        Parameters:
        - X_predict (pd.DataFrame): The new data to be projected. Must be a pandas DataFrame.

        Raises:
        - ValueError: If `X_predict` is not a pandas DataFrame.
        - ValueError: If the model has not been fitted yet.

        Returns:
        - hotelling_p2 (list of float): The Hotelling's T2 statistic for each observation in `X_predict`.
        - spe_p2 (list of float): The SPE statistic for each observation in `X_predict`.
        - residuals (np.ndarray): The residuals of the projection, indicating the difference between the original 
        data and its reconstruction from the principal components.
        - predicted_scores (pd.DataFrame): The scores of the projected data on the principal components.
        """
        if not isinstance(X_predict, pd.DataFrame):
            raise NotDataFrameError(type(X_predict).__name__)

        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
        hotelling_p2 = self.hotelling_t2(X_predict)
        
        predicted_scores = self.transform(X_predict)

        spe_p2, residuals = self.spe(X_predict)

        return hotelling_p2, spe_p2, residuals, predicted_scores
    
    def predict(self, X_predict):
        '''
        Predicts the probability of an observation being an outlier

        Parameters
        ----------
        X_predict : array-like, shape (n_samples, n_features)
            The data to be predicted

        Returns
        -------
        response : dict
            Dictionary containing the prediction of the model given an observation. It contains attributes such as the T2 and SPE limits and values, as well as the probability of the observation being an outlier
        '''
        if not isinstance(X_predict, pd.DataFrame):
            raise NotDataFrameError(type(X_predict).__name__)

        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
        hotelling, SPE, _, _ = self.project(X_predict)

        X_transform = X_predict.copy()
        if self._standardize:
            X_transform = self.preprocess(data=X_transform)

        dict_differences = {col: X_transform[col].values for col in X_transform.columns}
        # Hotelling's T2 control limit. Phase II
        dfn = self._ncomps
        dfd = self._nobs - self._ncomps
        const = (self._ncomps * (self._nobs**2 -1)) / (self._nobs * (self._nobs - self._ncomps))
        prob_hotelling = [float(1-f.cdf(np.array(value)/const, dfn, dfd)) for value in hotelling]
        
        # SPE control limit. Phase II
        b, nu = np.mean(self._spe), np.var(self._spe)
            
        df = (2*b**2)/nu
        const = nu/(2*b)

        prob_spe = [float(1-chi2.cdf(np.array(spe)/const, df)) for spe in SPE]

        spe_outlier = [bool(x) for x in (SPE >= self._spe_limit)]
        hotelling_outlier = [bool(x) for x in (hotelling >= self._hotelling_limit_p2)]

        ids = [str(idx) for idx in X_predict.index]

        response_json = {
                    'id': ids,
                    'hotelling': list(hotelling),
                    'spe': list(SPE),
                    'prob_hotelling': list(prob_hotelling),
                    'prob_spe': list(prob_spe),
                    'spe_outlier': spe_outlier,
                    'hotelling_outlier': hotelling_outlier,
                    'spe_ucl': float(self._spe_limit),
                    'hotelling_ucl': float(self._hotelling_limit_p2),
                    'differences': dict_differences
        }
        return response_json


    '''
    PLOTS
    '''
    def score_plot(self, comp1:int, comp2:int, hue:pd.Series=None, test_set:pd.DataFrame=None):
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
        if comp1 <= 0 or comp1>self._nvars or comp2 <= 0 or comp2>self._nvars:
            raise NComponentsError(self._nvars)
        
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
        scores = self._scores.reset_index()

        hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[12, 6]).encode(y='y').interactive()
        vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[12, 6]).encode(x='x').interactive()

        if hue is not None:
            # Check if hue is a python series or a numpy array
            if not isinstance(hue, pd.Series):
                raise TypeError("Hue must be a pandas Series")
            
            scores[hue.name] = hue
        
            scatter = alt.Chart(scores).mark_circle().encode(
                x=f"PC_{comp1}:Q",
                y=f"PC_{comp2}:Q",
                tooltip=[f"PC_{comp1}", f"PC_{comp2}", hue.name],
                color=alt.Color(hue.name)
            ).interactive()

        else:
            scatter = alt.Chart(scores).mark_point().encode(
                x=f"PC_{comp1}:Q",
                y=f"PC_{comp2}:Q",
                tooltip=[self._index_name, f"PC_{comp1}", f"PC_{comp2}"]
            ).interactive()

        if test_set is not None:

            scores_test = self.transform(test_set)

            scores_test = scores_test.reset_index()
            scatter_test = alt.Chart(scores_test).mark_point(color='black', opacity=.1).encode(
                x=f"PC_{comp1}",
                y=f"PC_{comp2}",
                tooltip=[self._index_name, f"PC_{comp1}", f"PC_{comp2}"]
            ).interactive()

            return (scatter_test+ scatter + vline + hline)

        return (scatter + vline + hline)
    
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
        
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()

        scores = self._scores.copy()
        if self._scores.shape[0]>5000:
            mask = np.random.choice(self._scores.shape[0], 5000, replace=False)
            scores = self._scores.iloc[mask]

        max_pc1 = self._scores[f'PC_{comp1}'].max()
        max_pc2 = self._scores[f'PC_{comp2}'].max()
        hypothenuse = (max_pc1**2 + max_pc2**2)**0.5

        max_loadings1 = self._loadings[f'PC_{comp1}'].max()
        max_loadings2 = self._loadings[f'PC_{comp2}'].max()
        hypothenuse_loadings = (max_loadings1**2 + max_loadings2**2)**0.5

        ratio = hypothenuse/hypothenuse_loadings

        loadings = self._loadings.copy()*ratio
        loadings.index.name = 'variable'
        loadings.reset_index(inplace=True)

        
        if hue is not None:
            # Check if hue is a python series or a numpy array
            if not isinstance(hue, pd.Series):
                raise ValueError("Hue must be a pandas Series")
            
            scores[hue.name] = hue

            scores_plot = alt.Chart(scores.reset_index()).mark_circle().encode(
                x=alt.X(f'PC_{comp1}',title=f'PC {comp1} - {self._explained_variance[comp1-1]*100:.2f} %'),
                y=alt.Y(f'PC_{comp2}',title=f'PC {comp2} - {self._explained_variance[comp2-1]*100:.2f} %'),
                tooltip=[self._index_name, f"PC_{comp1}", f"PC_{comp2}", hue.name],
                color=alt.Color(hue.name)
            ).interactive()

        else:
            scores_plot = alt.Chart(scores.reset_index()).mark_circle().encode(
                x=alt.X(f'PC_{comp1}',title=f'PC {comp1} - {self._explained_variance[comp1-1]*100:.2f} %'),
                y=alt.Y(f'PC_{comp2}',title=f'PC {comp2} - {self._explained_variance[comp2-1]*100:.2f} %'),
                tooltip=[self._index_name, f"PC_{comp1}", f"PC_{comp2}"]
            ).interactive()

        
        loadings_plot = alt.Chart(loadings).mark_circle(color='red').encode(
            x=f"PC_{comp1}",
            y=f"PC_{comp2}",
            tooltip=['variable', f"PC_{comp1}", f"PC_{comp2}"]
        )

        hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[12, 6]).encode(y='y')
        vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[12, 6]).encode(x='x')

        if test_set is not None:

            scores_test = self.transform(test_set)

            scores_test = scores_test.reset_index()
            scatter_test = alt.Chart(scores_test).mark_point(color='black', opacity=.1).encode(
                x=f"PC_{comp1}",
                y=f"PC_{comp2}",
                tooltip=[self._index_name, f"PC_{comp1}", f"PC_{comp2}"]
            ).interactive()

            return (scatter_test + scores_plot + loadings_plot + vline + hline)
    
        return (scores_plot + loadings_plot+ vline + hline)
    
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
        
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()

        loadings = self._loadings.copy()
        loadings.index.name = 'variable'
        loadings.reset_index(inplace=True)

        # Altair plot for the loadings
        return alt.Chart(loadings).mark_bar().encode(
            x=alt.X('variable', title='Variable'),
            y=alt.Y(f'PC_{comp}',title=f'Loadings of PC {comp} - {self._explained_variance[comp-1]*100:.2f} %'),
            tooltip=['variable', f'PC_{comp}']
        ).interactive()

    def difference_plot(self, 
                        X_obs:pd.Series,):   
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
        if not isinstance(X_obs, pd.DataFrame):
            raise NotDataFrameError(type(X_obs).__name__)
        
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
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
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
        hotelling = pd.DataFrame({'observation': range(self._nobs), 'T2': self._hotelling})

        hotelling_chart = alt.Chart(hotelling).mark_line().encode(
            x=alt.X('observation', title='Observation'),
            y=alt.Y('T2', title="Hotelling's T2"),
            tooltip=['observation', "T2"],
        ).properties(
            title=f'Hotelling\'s T2 statistic plot \n alpha: {self._alpha*100}% -- Threshold: {self._hotelling_limit_p1:.2f}',
        ).interactive()

        hotelling_chart.configure_title(
            fontSize=20,
            font='Courier',
            anchor='start',
            color='gray'
        )

        threshold = alt.Chart(
                        pd.DataFrame({'y': [self._hotelling_limit_p1]})).mark_rule(
                        strokeDash=[12, 6], color='red').encode(y='y')

        # Altair plot for the Hotelling's T2 statistic
        return (hotelling_chart + threshold)
    
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
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()

        if not isinstance(test_set, pd.DataFrame):
            raise NotDataFrameError(type(test_set).__name__)

        hotelling = self.hotelling_t2(test_set)

        n_obs = len(hotelling)

        hotelling = pd.DataFrame({'observation': range(n_obs), 'T2': hotelling})

        hotelling_chart = alt.Chart(hotelling).mark_line().encode(
            x=alt.X('observation', title='Observation'),
            y=alt.Y('T2', title="Hotelling's T2"),
            tooltip=['observation', "T2"],
        ).properties(
            title=f'Hotelling\'s T2 statistic plot \n alpha: {self._alpha*100}% -- Threshold: {self._hotelling_limit_p2:.2f}',
        ).interactive()

        hotelling_chart.configure_title(
            fontSize=20,
            font='Courier',
            anchor='start',
            color='gray'
        )

        threshold = alt.Chart(
                        pd.DataFrame({'y': [self._hotelling_limit_p2]})).mark_rule(
                        strokeDash=[12, 6], color='red').encode(y='y')

        # Altair plot for the Hotelling's T2 statistic
        return (hotelling_chart + threshold)

    
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

        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
        spe = pd.DataFrame({'observation': range(self._nobs), 'SPE': self._spe})

        spe_chart = alt.Chart(spe).mark_line().encode(
            x=alt.X('observation', title='Observation'),
            y=alt.Y('SPE', title='SPE'),
            tooltip=['observation', "SPE"],
        ).properties(
            title=f'SPE statistic plot \n alpha: {self._alpha*100}% -- Threshold: {self._spe_limit:.2f}',
        ).interactive()

        spe_chart.configure_title(
            fontSize=20,
            font='Courier',
            anchor='start',
            color='gray'
        )

        threshold = alt.Chart(
                        pd.DataFrame({'y': [self._spe_limit]})).mark_rule(
                        strokeDash=[12, 6], color='red').encode(y='y')

        # Altair plot for the SPE statistic
        return (spe_chart + threshold)
    
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
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
        
        if not isinstance(test_set, pd.DataFrame):
            raise NotDataFrameError(type(test_set).__name__)

        SPE, _ = self.spe(test_set)

        nobs = len(SPE)
        spe = pd.DataFrame({'observation': range(nobs), 'SPE': SPE})

        spe_chart = alt.Chart(spe).mark_line().encode(
            x=alt.X('observation', title='Observation'),
            y=alt.Y('SPE', title='SPE'),
            tooltip=['observation', "SPE"],
        ).properties(
            title=f'SPE statistic plot \n alpha: {self._alpha*100}% -- Threshold: {self._spe_limit:.2f}',
        ).interactive()

        spe_chart.configure_title(
            fontSize=20,
            font='Courier',
            anchor='start',
            color='gray'
        )

        threshold = alt.Chart(
                        pd.DataFrame({'y': [self._spe_limit]})).mark_rule(
                        strokeDash=[12, 6], color='red').encode(y='y')

        # Altair plot for the SPE statistic
        return (spe_chart + threshold)

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
        if not hasattr(self, '_scores'):
            raise ModelNotFittedError()
       
        if not isinstance(data, pd.DataFrame):
            raise NotDataFrameError(type(data).__name__)
    
        if data.shape[1] != len(self._variables):
            raise ValueError(f'Number of features in data must be {len(self._variables)}')
        
        if data.shape[0] != 1:
            raise ValueError(f'Number of observations in data must be 1')
        
        SPE, residuals = self.spe(data)
        

        residuals = pd.DataFrame({'variable': self._variables, 'residual': residuals[0]})
        # Altair plot for the residuals
        return alt.Chart(residuals).mark_bar().encode(
            x=alt.X('variable', title='Variable'),
            y=alt.Y('residual', title='Residual'),
            tooltip=['variable', 'residual']
        ).properties(
            title=f'Residuals for observation {str(data.index.values[0])} - SPE: {SPE[0]:.2f}'
        ).interactive()