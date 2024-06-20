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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, n_comps=None, standardize=True, tolerance=1e-4, verbose=False):
        
        if not 0 < tolerance < 1:
            raise ValueError('Tolerance must be strictly between 0 and 1')

        self._standardize = standardize
        self._tolerance = tolerance
        self.verbose = verbose
        self._ncomps = n_comps

        self._scaler = Pipeline([
            # ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

    def get_params(self, deep=True):
        return {"standardize": self._standardize,
                "tolerance": self._tolerance,
                "verbose": self._verbose,
                "n_comps": self._ncomps}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


    def train(self, data, numerical_features=[], alpha=0.95):

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be of type pandas DataFrame, not {type(data)}')

        if not isinstance(numerical_features, list):
            raise ValueError(f'Numerical features must be a list, not {type(numerical_features)}')
        
        if not 0 < alpha < 1:
            raise ValueError('Alpha must be strictly between 0 and 1')

        self._alpha = alpha
        self.fit(data, numerical_features)

        self.spe()
        self.hotelling_t2()

        self.control_limits(alpha=self._alpha)

            
    def fit(self, data, numerical_features=[]):
        '''
        Fits the PCA model to the data

        Parameters
        ----------
        data: array-like, shape (n_samples, n_features)
            The data to be fitted

        Returns
        -------
        None
        '''
        if self._ncomps==None:
            self._ncomps = data.shape[1]

        if self._ncomps <= 0 or self._ncomps>data.shape[1]:
            raise ValueError(f'The number of components must be between 0 and {data.shape[1]}')
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be of type pandas DataFrame, not {type(data)}')
        
        if not isinstance(numerical_features, list):
            raise ValueError(f'Numerical features must be a list, not {type(numerical_features)}')
        
        self._variables = data.columns
        self._index = data.index       
        self._index_name = data.index.name
        self._nobs, self._nvars = data.shape

        self._numerical_features = numerical_features

        X = data.copy()
        # Descale and demean matrix
        if self._standardize:
            if self._numerical_features:
                X[self._numerical_features] = self._scaler.fit_transform(X[self._numerical_features])
            else:
                X = pd.DataFrame(self._scaler.fit_transform(X), columns=self._variables, index=self._index)
        
        X = X.values

        X_pca = X.copy()
        r2 = []
        T = np.zeros((self._ncomps, X.shape[0]))
        P_t = np.zeros((self._ncomps, X.shape[1]))
        vals = np.zeros(self._ncomps)

        for i in range(self._ncomps):
            # Initialize t as the column with the highest variance
            column = np.argmax(np.var(X_pca, axis=0))
            t = X_pca[:, column].reshape(-1, 1)
            cont = 0
            conv = 10

            while conv > self._tolerance:
                t_prev = t
                p_t = (t_prev.T @ X_pca) / (t_prev.T @ t_prev)
                p_t /= LA.norm(p_t)

                t = X_pca @ p_t.T

                conv = np.linalg.norm(t - t_prev)
                cont += 1

            if self.verbose:
                print(f"Component {i+1} converges after {cont} iterations")

            X_pca -= t @ p_t  # Calculate the residual matrix
            r2.append(1 - np.sum(X_pca**2) / np.sum(X**2))

            vals[i] = np.var(t)
            T[i] = t.reshape(X.shape[0])
            P_t[i] = p_t

        self._eigenvals = vals
        self._loadings = pd.DataFrame(P_t, columns=self._variables, index=[f"PC_{i}" for i in range(1, self._ncomps+1)])
        self._training_data = data
        self._processed_training_data = X
        self._scores = pd.DataFrame(T.T, columns=[f"PC_{i}" for i in range(1, self._ncomps+1)], index=self._index)
        self._rsquared_acc = np.array(r2)
        self._explained_variance = np.diff(np.insert(self._rsquared_acc, 0, 0))
        self._residuals_fit = X-T.T@P_t
        self._mean_train = X.mean()
        self._std_train = X.std()

    def transform(self, data, y=None):
        '''
        Projects a set of data onto the PCA space

        Parameters
        ----------
        data: array-like, shape (n_samples, n_features)
            The data to be projected onto the PCA space

        y: None
            This is only added so we can use this method within a scikit-learn pipeline

        Returns
        -------
        array-like, shape (n_samples, n_components)
            The projected data
        '''

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be of type pandas DataFrame, not {type(data)}')

        X_transform = data.copy()
        # Descale and demean matrix
        if self._standardize:
            if self._numerical_features:
                X_transform[self._numerical_features] = self._scaler.transform(X_transform[self._numerical_features])
            else:
                X_transform = pd.DataFrame(self._scaler.transform(X_transform), columns=self._variables, index=self._index)

        self._scores_test = pd.DataFrame(X_transform @ self._loadings.T, columns=self._scores.columns, index=data.index)
        
        return self._scores_test 

    def fit_transform(self, data, y=None):
        '''
        Fits the PCA model and projects the data onto the PCA space

        Parameters
        ----------
        data: array-like, shape (n_samples, n_features)
            The data to be projected onto the PCA space

        y: None
            This is only added so we can use this method within a scikit-learn pipeline
        Returns
        -------
        array-like, shape (n_samples, n_components)
            The projected data
        '''
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        '''
        Projects the data back to the original space

        Parameters
        ----------
        data: array-like, shape (n_samples, n_components)
            The data to be projected back to the original space

        Returns
        -------
        array-like, shape (n_samples, n_features)
            The projected data
        '''

        if isinstance(data, pd.DataFrame):
            data = data.values

        return data @ self._loadings
    
    def control_limits(self, alpha:float=0.95):
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
    
    def hotelling_t2(self, alpha:float=0.95):
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

        Parameters
        ----------
        alpha : float
            Type I error. 1-alpha is the probability of rejecting a true null hypothesis.

        Returns
        -------
        Hotelling's T2 statistic for every observation.
        '''

        self._hotelling = np.array([np.sum((self._scores.values[i, :]**2)/self._eigenvals) 
                                    for i in range(self._nobs)])
        
    def spe(self):
        '''
        Represents the sum of squared prediction errors. The value is given by the expression 
        e^T_i * e_i, so the SPE statistic is the scalar product of the residuals vector of observation i
        multiplied by its transposed self.
        
        The SPE statistic represents the squared Euclidean distance of an observation from the generated
        subspace, and gives a measure of how close the observation is to the A-dimensional subspace.
        
        Observations that are above the control limit can be considered as moderate outliers.

        Parameters
        ----------
        alpha : float
            Type I error. 1-alpha is the probability of rejecting a true null hypothesis.
        plot : bool, optional
            If True, a plot of the Hotelling's T2 statistic and the control limit will be displayed. The default is True.

        Returns
        -------
        SPE statistic for every observation.

        '''
        
        self._spe = np.array([self._residuals_fit[i, :].T@self._residuals_fit[i, :] for i in range(self._nobs)])

    def spe_p2(self, alpha:float=0.95):
        '''
        Represents the sum of squared prediction errors. The value is given by the expression 
        e^T_i * e_i, so the SPE statistic is the scalar product of the residuals vector of observation i
        '''

        SPE = np.array([self._residuals_fit[i, :].T@self._residuals_fit[i, :] for i in range(self._nobs)])

        b, nu = np.mean(SPE), np.var(SPE)
        
        df = (2*b**2)/nu
        const = nu/(2*b)

        self._spe = SPE
        self._spe_limit_p1 = chi2.ppf(alpha, df)*const

    def hotelling_t2_phase2(self, data):

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be of type pandas DataFrame, not {type(data)}')

        X = data.copy()

        # Descale and demean matrix
        if self._standardize:
            if self._numerical_features:
                X[self._numerical_features] = self._scaler.transform(X[self._numerical_features])
            else:
                X = pd.DataFrame(self._scaler.transform(X), columns=self._variables, index=self._index)

        X = X.values

        T2 = np.array([np.sum((X[i, :] @ self._loadings.values)**2/self._eigenvals) for i in range(X.shape[0])])

        return T2

    def predict(self, X_predict):

        if not isinstance(X_predict, pd.DataFrame):
            raise ValueError(f'Data must be of type pandas DataFrame, not {type(X_predict)}')

        if not hasattr(self, '_scores'):
            raise ValueError("The model has not been fitted yet. Please use the fit method before predicting")
        
        predicted_scores = self.transform(X_predict)

        X_transform = X_predict.copy()
        
        if self._standardize:
            if self._numerical_features:
                X_transform[self._numerical_features] = self._scaler.transform(X_transform[self._numerical_features])
            else:
                X_transform = pd.DataFrame(self._scaler.transform(X_transform), columns=self._variables, index=self._index)

        # Hotelling's T2 statistic
        self._hotelling_p2 = np.array([np.sum((predicted_scores.values[i, :]**2)/self._eigenvals) 
                                    for i in range(predicted_scores.shape[0])])
        
        # SPE statistic
        residuals = X_transform.values - predicted_scores.values @ self._loadings.values

        self._spe_p2 = np.array([residuals[i, :].T@residuals[i, :] for i in range(predicted_scores.shape[0])])

        return self._hotelling_p2, self._spe_p2

    '''
    PLOTS
    '''
    def score_plot(self, comp1:int, comp2:int, hue:pd.Series=None, plot_test=False):
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
        if comp1 <= 0 or comp2 <= 0:
            raise ValueError("The number of components must be greather than 0")
        
        if plot_test==True and not hasattr(self, '_scores_test'):
            raise ValueError("No test data has been projected onto the latent space. Please use the transform or the fit_transform method before plotting the test data")

        scores = self._scores.reset_index()

        hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[12, 6]).encode(y='y').interactive()
        vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[12, 6]).encode(x='x').interactive()

        if hue is not None:
            # Check if hue is a python series or a numpy array
            if not isinstance(hue, pd.Series):
                raise ValueError("Hue must be a pandas Series")
            
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

        if plot_test:
            scores_test = self._scores_test.reset_index()
            scatter_test = alt.Chart(scores_test).mark_point(color='black', opacity=.1).encode(
                x=f"PC_{comp1}",
                y=f"PC_{comp2}",
                tooltip=[self._index_name, f"PC_{comp1}", f"PC_{comp2}"]
            ).interactive()

            return (scatter_test+ scatter + vline + hline)

        return (scatter + vline + hline)
    
    def biplot(self, comp1:int, comp2:int, hue:pd.Series=None, plot_test=False):
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
        plot_test : bool
            If True, the test data will be plotted as well

        Returns
        -------
        None
        '''
        if comp1 <= 0 or comp2 <= 0:
            raise ValueError("The number of components must be greather than 0")
        
        if plot_test==True and not hasattr(self, '_scores_test'):
            raise ValueError("No test data has been projected onto the latent space. Please use the transform or the fit_transform method before plotting the test data")

        scores = self._scores.copy()
        if self._scores.shape[0]>5000:
            mask = np.random.choice(self._scores.shape[0], 5000, replace=False)
            scores = self._scores.iloc[mask]

        max_pc1 = self._scores[f'PC_{comp1}'].max()
        max_pc2 = self._scores[f'PC_{comp2}'].max()
        hypothenuse = (max_pc1**2 + max_pc2**2)**0.5

        max_loadings1 = self._loadings.T[f'PC_{comp1}'].max()
        max_loadings2 = self._loadings.T[f'PC_{comp2}'].max()
        hypothenuse_loadings = (max_loadings1**2 + max_loadings2**2)**0.5

        ratio = hypothenuse/hypothenuse_loadings

        loadings = self._loadings.T.copy()*ratio
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

        if plot_test:
            scores_test = self._scores_test.reset_index()
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
        if comp <= 0:
            raise ValueError("The number of components must be greather than 0")

        loadings = self._loadings.T.copy()
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
        Generates a bar plot of the difference between a specific observation and the mean of the sample

        Parameters
        ----------
        obs : int
            The number of the observation.

        Returns
        -------
        None
        '''
        if not isinstance(X_obs, pd.DataFrame):
            raise ValueError("The observation must be a pandas Series or a DataFrame")
        
        if sorted(X_obs.columns) != sorted(self._variables):
            raise ValueError("The observation must have the same variables as the training data")
                
        if self._standardize:
            if self._numerical_features:
                X_obs[self._numerical_features] = self._scaler.transform(X_obs[self._numerical_features])
            else:
                X_obs = pd.DataFrame(self._scaler.transform(X_obs), columns=self._variables, index=X_obs._index)

        df_plot = pd.DataFrame({'variable': self._variables, 'value': X_obs.values[0]})
        # Altair plot for the differences
        return alt.Chart(df_plot).mark_bar().encode(
            x=alt.X('variable', title='Variable'),
            y=alt.Y('value', title='Difference with respect to the mean (std)'),
            tooltip=['variable', 'value']
        ).interactive()

    def hotelling_t2_plot_p1(self):
        '''
        Generates a plot of the Hotelling's T2 statistic

        Parameters
        ----------
        alpha : float
            Type I error. 1-alpha is the probability of rejecting a true null hypothesis.

        Returns
        -------
        None
        '''
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
    
    def hotelling_t2_plot_p2(self):
        '''
        Generates a plot of the Hotelling's T2 statistic for Phase II

        '''

        n_obs = self._scores_test.shape[0]  

        if not hasattr(self, '_hotelling_p2'):
            raise ValueError("No test data has been projected onto the latent space. Please use the predict() method before plotting the test data")
        
        hotelling = pd.DataFrame({'observation': range(n_obs), 'T2': self._hotelling_p2})

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
        Generates a plot of the SPE statistic

        Parameters
        ----------
        alpha : float
            Type I error. 1-alpha is the probability of rejecting a true null hypothesis.

        Returns
        -------
        None
        '''
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
    
    def spe_plot_p2(self):
        '''
        Generates a plot of the SPE statistic

        Parameters
        ----------
        alpha : float
            Type I error. 1-alpha is the probability of rejecting a true null hypothesis.

        Returns
        -------
        None
        '''

        nobs = self._scores_test.shape[0]
        spe = pd.DataFrame({'observation': range(nobs), 'SPE': self._spe_p2})

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

    def residual_barplot(self, obs:int):
        '''
        Generates a bar plot of the residuals of the selected observation

        Parameters
        ----------
        obs : int
            The number of the observation.

        Returns
        -------
        None
        '''
        if obs < 0 or obs >= self._nobs:
            raise ValueError("The observation number must be between 0 and the number of observations")

        residuals = pd.DataFrame({'variable': self._variables, 'residual': self._residuals_fit[obs]})
        # Altair plot for the residuals
        return alt.Chart(residuals).mark_bar().encode(
            x=alt.X('variable', title='Variable'),
            y=alt.Y('residual', title='Residual'),
            tooltip=['variable', 'residual']
        ).properties(
            title=f'Residuals for observation {obs} - SPE: {self._spe[obs]:.2f}'
        ).interactive()
    
    def spe_contribution_plot(self, obs:int):
        '''
        Generates a bar plot of the contribution of each variable to the SPE statistic of the selected observation

        Parameters
        ----------
        obs : int
            The number of the observation.

        Returns
        -------
        None
        '''
        if obs < 0 or obs >= self._nobs:
            raise ValueError("The observation number must be between 0 and the number of observations")

        residuals = self._residuals_fit[obs]**2
        residuals = pd.DataFrame({'variable': self._variables, 'contribution': residuals})

        # Altair plot for the residuals
        return alt.Chart(residuals).mark_bar().encode(
            x=alt.X('variable', title='Variable'),
            y=alt.Y('contribution', title='Contribution'),
            tooltip=['variable', 'contribution']
        ).properties(
            title=f'Contribution to the SPE of observation {obs} - SPE: {self._spe[obs]:.2f}'
        ).interactive()
    
    def hotelling_t2_contribution_plot(self, obs:int):
        '''
        Generates a bar plot of the contribution of each variable to the Hotelling's T2 statistic of the selected observation

        Parameters
        ----------
        obs : int
            The number of the observation.

        Returns
        -------
        None
        '''
        if obs < 0 or obs >= self._nobs:
            raise ValueError("The observation number must be between 0 and the number of observations")

        contributions = (self._loadings.values*self._processed_training_data[obs])
        normalized_contributions = (contributions/self._eigenvals[:, None])**2

        max_comp = np.argmax(np.sum(normalized_contributions, axis=1))

        contributions_df = pd.DataFrame({'variable': self._variables, 'contribution': contributions[max_comp]})

        # Altair plot for the residuals
        return alt.Chart(contributions_df).mark_bar().encode(
            x=alt.X('variable', title='Variable'),
            y=alt.Y('contribution', title='Contribution'),
            tooltip=['variable', 'contribution']
        ).properties(
            title=f'Contribution to the Hotelling\'s T2 of observation {obs} - T2: {self._hotelling[obs]:.2f} - Comp: {max_comp}'
        ).interactive()