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

class PCA:
    def __init__(self, demean=True, descale=True, tolerance=1e-4, verbose=False):
        
        if not 0 < tolerance < 1:
            raise ValueError('Tolerance must be strictly between 0 and 1')

        self._demean = demean
        self._descale = descale
        self._tolerance = tolerance
        self.verbose = verbose
            
    def fit(self, data, ncomps=None):
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
        if ncomps <= 0 or ncomps>data.shape[1]:
            raise ValueError(f'The number of components must be between 0 and {data.shape[1]}')
    
        if ncomps==None:
            ncomps = data.shape[1]

        self._ncomps = ncomps

        #Check if X_train is a pandas dataframe
        if isinstance(data, pd.DataFrame):
            self._variables = data.columns
            data = data.values
        else:
            self._variables = [f"X{i}" for i in range(data.shape[1])]
        
        self._nobs, self._nvars = data.shape
        X = data.copy()

        # Descale and demean matrix
        if self._demean:
            X -= np.mean(X, axis=0)
        if self._descale:
            X /= np.std(X, axis=0)

        r2 = []
        T = np.zeros((self._ncomps, X.shape[0]))
        P_t = np.zeros((self._ncomps, X.shape[1]))
        vals = np.zeros(self._ncomps)

        for i in range(self._ncomps):
            # Initialize t as the column with the highest variance
            column = np.argmax(np.var(X, axis=0))
            t = X[:, column].reshape(-1, 1)
            cont = 0
            conv = 10

            while conv > self._tolerance:
                t_prev = t
                p_t = (t_prev.T @ X) / (t_prev.T @ t_prev)
                p_t /= LA.norm(p_t)

                t = X @ p_t.T

                conv = np.linalg.norm(t - t_prev)
                cont += 1

            if self.verbose:
                print(f"Component {i+1} converges after {cont} iterations")

            X -= t @ p_t  # Calculate the residual matrix
            r2.append(1 - np.sum(X**2) / np.sum(data**2))

            vals[i] = np.var(t)
            T[i] = t.reshape(X.shape[0])
            P_t[i] = p_t

        self._eigenvals = vals
        self._loadings = pd.DataFrame(P_t, columns=self._variables, index=[f"PC_{i}" for i in range(1, self._ncomps+1)])
        self._training_data = data
        self._scores = pd.DataFrame(T.T, columns=[f"PC_{i}" for i in range(1, self._ncomps+1)])
        self._rsquared_acc = np.array(r2)
        self._explained_variance = np.diff(np.insert(self._rsquared_acc, 0, 0))
        self._residuals_fit = data-T.T@P_t
        self._mean_train = data.mean()
        self._std_train = data.std()

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

        if isinstance(data, pd.DataFrame):
            data = data.values

        if self._demean:
            data -= self._mean_train
        if self._descale:
            data /= self._std_train
        
        return pd.DataFrame(data @ self._loadings.T, columns=self._scores.columns)

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

        dfn = self._ncomps/2
        dfd = (self._nobs-self._ncomps-1)/2
        const = ((self._nobs-1)**2)/self._nobs

        self._hotelling = np.array([np.sum((self._scores.values[i, :]**2)/self._eigenvals) for i in range(self._nobs)])
        self._hotelling_limit = beta.ppf(alpha, dfn, dfd)*const
        
    def spe(self, alpha:float=0.95):
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
        
        SPE = np.array([self._residuals_fit[i, :].T@self._residuals_fit[i, :] for i in range(self._nobs)])

        b, nu = np.mean(SPE), np.var(SPE)
        
        df = (2*b**2)/nu
        const = nu/(2*b)

        self._spe = SPE
        self._spe_limit = chi2.ppf(alpha, df)*const
    '''
    PLOTS
    '''
    def score_plot(self, comp1:int, comp2:int):
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

        scores = self._scores.copy()
        if self._scores.shape[0]>5000:
            mask = np.random.choice(self._scores.shape[0], 5000, replace=False)
            scores = self._scores.iloc[mask]

        # Altair plot for the scores. Includes a horizontal and a vertical line at 0
        scatter = alt.Chart(scores).mark_circle().encode(
            x=f"PC_{comp1}",
            y=f"PC_{comp2}",
            tooltip=[f"PC_{comp1}", f"PC_{comp2}"]
        ).interactive()

        hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[12, 6]).encode(y='y')
        vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[12, 6]).encode(x='x')
    
        return (scatter + vline + hline)
    
    def biplot(self, comp1:int, comp2:int):
        '''
        Generates a scatter plot of the selected components with the scores and the loadings

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

        scores = self._scores.copy()
        if self._scores.shape[0]>5000:
            mask = np.random.choice(self._scores.shape[0], 5000, replace=False)
            scores = self._scores.iloc[mask]

        max_pc1 = self._scores[f'PC_{comp1}'].max()
        max_pc2 = self._scores[f'PC_{comp2}'].max()
        hypothenuse = (max_pc1**2 + max_pc2**2)**0.5
        hypothenuse

        max_loadings1 = self._loadings.T[f'PC_{comp1}'].max()
        max_loadings2 = self._loadings.T[f'PC_{comp2}'].max()
        hypothenuse_loadings = (max_loadings1**2 + max_loadings2**2)**0.5

        ratio = hypothenuse/hypothenuse_loadings

        loadings = self._loadings.T.copy()*ratio
        loadings.index.name = 'variable'
        loadings.reset_index(inplace=True)

        # Altair plot for the scores. Includes a horizontal and a vertical line at 0
        scores_plot = alt.Chart(scores.reset_index()).mark_circle().encode(
            x=alt.X(f'PC_{comp1}',title=f'PC {comp1} - {self._explained_variance[comp1-1]*100:.2f} %'),
            y=alt.Y(f'PC_{comp2}',title=f'PC {comp2} - {self._explained_variance[comp2-1]*100:.2f} %'),
            tooltip=['index', f"PC_{comp1}", f"PC_{comp2}"]
        ).interactive()

        loadings_plot = alt.Chart(loadings).mark_circle(color='red').encode(
            x=f"PC_{comp1}",
            y=f"PC_{comp2}",
            tooltip=['variable', f"PC_{comp1}", f"PC_{comp2}"]
        )

        hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[12, 6]).encode(y='y')
        vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[12, 6]).encode(x='x')
    
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

    def difference_plot(self, obs:int):
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
        if obs < 0 or obs >= self._nobs:
            raise ValueError("The observation number must be between 0 and the number of observations")

        standardized_observation = (self._training_data[obs] - self._mean_train) / self._std_train

        df_observation = pd.DataFrame({'variable': self._variables, 'value': standardized_observation})

        # Altair plot for the differences
        return alt.Chart(df_observation).mark_bar().encode(
            x=alt.X('variable', title='Variable'),
            y=alt.Y('value', title='Difference with respect to the mean (std)'),
            tooltip=['variable', 'value']
        ).interactive()

    def hotelling_t2_plot(self, alpha:float=0.95):
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
        # if not hasattr(self, '_hotelling'):
        self.hotelling_t2(alpha)

        hotelling = pd.DataFrame({'observation': range(self._nobs), 'T2': self._hotelling})

        hotelling_chart = alt.Chart(hotelling).mark_line().encode(
            x=alt.X('observation', title='Observation'),
            y=alt.Y('T2', title="Hotelling's T2"),
            tooltip=['observation', "T2"],
        ).properties(
            title=f'Hotelling\'s T2 statistic plot \n alpha: {alpha*100}% -- Threshold: {self._hotelling_limit:.2f}',
        ).interactive()

        hotelling_chart.configure_title(
            fontSize=20,
            font='Courier',
            anchor='start',
            color='gray'
        )

        threshold = alt.Chart(
                        pd.DataFrame({'y': [self._hotelling_limit]})).mark_rule(
                        strokeDash=[12, 6], color='red').encode(y='y')

        # Altair plot for the Hotelling's T2 statistic
        return (hotelling_chart + threshold)
    
    def spe_plot(self, alpha:float=0.95):
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
        # if not hasattr(self, '_spe'):
        self.spe(alpha)

        spe = pd.DataFrame({'observation': range(self._nobs), 'SPE': self._spe})

        spe_chart = alt.Chart(spe).mark_line().encode(
            x=alt.X('observation', title='Observation'),
            y=alt.Y('SPE', title='SPE'),
            tooltip=['observation', "SPE"],
        ).properties(
            title=f'SPE statistic plot \n alpha: {alpha*100}% -- Threshold: {self._spe_limit:.2f}',
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

        contributions = (self._loadings.values*self._training_data[obs])
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
    

    
# class PCR(PCA):
#     def __init__(self, X, ncomps, autoescalado = True, tolerancia = 1e-15, verbose = False):
#         PCA.__init__(self, X, ncomps, autoescalado = True, tolerancia = 1e-15, verbose = False)
#         self.rsquared_fit = None
#         self.rsquared_pred = None
#         self.coefs = None
#         self.ssr_fit = None
#         self.press = None
#         self.prediction = None
    
    
#     def fit(self, y_train):
        
#         self.y_train = y_train
#         y_train = np.asarray(y_train)
        
#         r2_PCR = []
#         T_PCR, P_PCR = self.scores, self.loadings
        
#         X_train = self.X

#         b = np.linalg.inv(np.transpose(T_PCR).dot(T_PCR)).dot(np.transpose(T_PCR)).dot(y_train)
        
#         B_PCR = np.transpose(P_PCR).dot(b)
        
#         y_hat_PCR = X_train.dot(B_PCR)
        
#         r2_PCR = 100*(1- np.sum((y_train-y_hat_PCR)**2)/np.sum((y_train-np.mean(y))**2))
    
#         self.rsquared_fit = r2_PCR
#         self.coefs = B_PCR
#         self.ssr_fit = np.sum((y_train-y_hat_PCR)**2)
        
#     def predict(self, X_test, y_test):
        
#         B_PCR = self.coefs
        
#         y_prediction = X_test.dot(B_PCR)
        
#         self.prediction = y_prediction
#         r2_pred = 100*(1- np.sum((y_test-y_prediction)**2)/np.sum((y_test-np.mean(y_test))**2))
#         self.rsquared_pred = r2_pred
#         self.press = np.sum((y_prediction-y_test)**2)

# class PLS(object):
#     def __init__(self, X,y,ncomps, tol=1e-15, autoescalado=True):
        
#         self.X = np.asarray(X)
#         self.y = np.asarray(y)
#         self._ncomps = ncomps
#         self._autoesc = autoescalado
#         self._tolerancia = tol
#         self._nobs, self._nvars = self.X.shape
        
#         self.T = None
#         self.P_t = None
#         self.U = None
#         self.C_t = None
#         self.W = None
        
#         self.rsquare_X = None
#         self.rsquare_y = None
        
        
#     def nipals(self, X, y, n_componentes, autoesc=True):
#         X_original = self.X
#         X = self.X
        
#         y_original = self.y
#         y=self.y
        
#         dif = self._tolerancia
            
        
#         #Establecemos la posibilidad de autoescalar. Por defecto, la función autoescalará
#         if self._autoesc==True:
#             for i in range(X.shape[1]):
#                 X[:,i]= X[:,i]-np.mean(X[:,i])
#                 X[:,i]= X[:,i]/np.std(X[:,i])
                
#             for i in range(y.shape[1]):   
#                 y[:,i]= y[:,i]-np.mean(y[:,i])
#                 y[:,i]= y[:,i]/np.std(y[:,i])
        
        
#         if not 0 < self._tolerancia < 1:
#             raise ValueError('Tolerance must be strictly between 0 and 1')
                
#         print("********* Algoritmo NIPALS para PLS ***********")
#         #Inicializamos las matrices de scores y de loadings según el número de componentes propuesto
#         r2_X = []
#         T = np.zeros(shape=(self._ncomps, X.shape[0]))
#         P_t = np.zeros(shape = (self._ncomps, X.shape[1]))
        
#         r2_y = []
#         U = np.zeros(shape=(self._ncomps, y.shape[0]))
#         C_t = np.zeros(shape = (self._ncomps, y.shape[1]))
        
#         W_t = np.zeros(shape = (self._ncomps, y.shape[1]))
        
        
#         for i in range(self._ncomps):
            
#             #Iniciamos u como la primera columna de Y
#             u = np.array(y[:,0])
#             u.shape=(y.shape[0], 1) #Esto sirve para obligar a que t sea un vector columna
            
#             cont=0
#             conv=0
            
#             while conv<y.shape[0]:
#                 u_previo = u
#                 w_t = (np.transpose(u_previo).dot(X))/(np.transpose(u_previo).dot(u_previo))
#                 w_t = w_t/LA.norm(w_t)
                
#                 t=X.dot(np.transpose(w_t))
                
#                 c_t = np.transpose(t).dot(y)/(np.transpose(t).dot(t))
                
#                 u = y.dot(np.transpose(c_t))/(c_t.dot(np.transpose(c_t)))
#                 conv = np.sum((u-u_previo)<dif)
#                 cont+=1
                
#             p_t=np.transpose(t).dot(X)/(np.transpose(t).dot(t))
            
#             print("Componente ", i+1, " converge en ", cont, " iteraciones")
#             E = X-t.dot(p_t)
#             F=y-t.dot(c_t)
            
#             r2_X.append(1-np.sum(E**2)/np.sum(X_original**2))
#             r2_y.append(1-np.sum(F**2)/np.sum(y_original**2))
            
#             X=E
#             Y=F
            
#             T[i]=t.reshape((X.shape[0]))
#             P_t[i]=p_t
            
#             U[i]=u.reshape((X.shape[0]))
#             C_t[i]=c_t
            
#             W_t[i]= w_t
            
#         T = np.transpose(T)
#         U = np.transpose(U)
        
#         self.T = T
#         self.P_t = P_t
#         self.U = U
#         self.C_t = C_t
#         self.W = W_t
        
#         self.rsquare_X = r2_X
#         self.rsquare_y = r2_Y
    

# def optimize_SPE(X_train, ncomps, alpha, threshold, iterations=500, tol=1e-15):
#     limit_SPE=1000
#     highest=1000
#     tam = X_train.shape[0]
    
#     while highest > 0:
#         model = PCA(tolerancia=tol)
#         model.fit(X_train, ncomps)
            
#         T = model.scores
#         P_t = model.loadings
#         E=X_train-T.dot(P_t)
                
#         obs = X_train.shape[0]
    
#         spe = np.array([np.transpose(E[i,:]).dot(E[i,:]) for i in range(E.shape[0])])
#         b = np.mean(spe)
#         nu = np.var(spe)
    
#         ucl_SPE = nu/(2*b)*chi2.ppf(alpha, (2*b**2)/nu)
        
#         greater = []
#         for k in range(obs):
#             if spe[k]>threshold*ucl_SPE:
#                 greater.append(k)
                
#         if max(spe)>threshold*ucl_SPE:
#             X_train = np.delete(X_train,np.where(spe ==max(spe)),0)
    
#         highest = len(greater) 
    
    
#     while limit_SPE > (1-alpha)*tam:

        
#         model = PCA(tolerancia=tol)
#         model.fit(X_train, ncomps)
        
#         T = model.scores
#         P_t = model.loadings
#         E=X_train-T.dot(P_t)
                
#         obs = X_train.shape[0]

#         spe = np.array([np.transpose(E[i,:]).dot(E[i,:]) for i in range(E.shape[0])])
#         b = np.mean(spe)
#         nu = np.var(spe)
    
#         ucl_SPE = nu/(2*b)*chi2.ppf(alpha, (2*b**2)/nu)
        
#         greater = []
#         for k in range(obs):
#             if spe[k]>ucl_SPE:
#                 greater.append(k)
                
#         if max(spe)>ucl_SPE:
#                 k = np.where(spe ==max(spe))
#                 X_train = np.delete(X_train, k, 0)      
                
#         limit_SPE= len(greater)

#     model= PCA(tolerancia = tol, autoescalado = False)
#     X_opt = X_train
#     model.fit(X_opt, ncomps)    
#     return(X_opt, model)
    
# def optimize_T2(X_train, ncomps, alpha, threshold, iterations=10, tol=1e-15):
#     limit_T2=1000
#     tam = X_train.shape[0]
#     highest = 1000
    
#     while highest > 0:
#         model = PCA(tolerancia=tol)
#         model.fit(X_train, ncomps)
            
#         T = model.scores
#         P_t = model.loadings
                
#         obs = X_train.shape[0]
        
#         dfn = ncomps/2
#         dfd = (obs-ncomps-1)/2
#         const = ((obs-1)**2)/obs
#         tau = np.array([np.sum(((T[i])**2)/np.var(T[i])) for i in range(obs)])
    
#         ucl_T2 = (beta.ppf(alpha, dfn, dfd))*const
        
#         greater = []
#         for k in range(obs):
#             if tau[k]>threshold*ucl_T2:
#                 greater.append(k)
                
#         if max(tau)>threshold*ucl_T2:
#             X_train = np.delete(X_train,np.where(tau ==max(tau)),0)
    
#         highest = len(greater)

#     while limit_T2 > (1-alpha)*tam:
        
#         model = PCA(tolerancia = tol)
#         model.fit(X_train, ncomps)
        
#         T = model.scores
        
#         obs = X_train.shape[0]
        
#         dfn = ncomps/2
#         dfd = (obs-ncomps-1)/2
#         const = ((obs-1)**2)/obs
#         tau = np.array([np.sum(((T[i])**2)/np.var(T[i])) for i in range(obs)])
    
#         ucl_T2 = (beta.ppf(alpha, dfn, dfd))*const
        
#         greater = []
#         for k in range(X_train.shape[0]):
#             if tau[k]>ucl_T2:
#                 greater.append(k)
                
#         if max(tau)>ucl_T2:
#                 l = np.where(tau ==max(tau))
#                 X_train = np.delete(X_train, l, 0)

#         limit_T2= len(greater)
        

#     model= PCA(tolerancia = tol, autoescalado = False)
#     X_opt = X_train
#     model.fit(X_opt, ncomps)
    
#     return(X_opt, model)