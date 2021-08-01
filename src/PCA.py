import numpy as np
from utils import nipals
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, chi2


np.seterr(divide='ignore', invalid='ignore')
sns.set_style("darkgrid")

class PCA():
    def __init__(self, X):
        
        self._nobs = X.shape[0]
        self._nvars = X.shape[1]
        self._X = X

    def fit(self, ncomps=None, threshold=1e-12, demean=True, standardize=True, verbose=True, max_iterations=10000):
        '''
        Fits a PCA model to the provided data using the NIPALS algorithm

        Parameters
        ----------
        ncomps : int
            Number of components to fit. If no argument is provided, the algorithm will take
            as many components as columns
        threshold : float, optional
            Threshold for the convergence of the algorithm. The default is 1e-12.
        demean : bool, optional
            If True, every column will have their mean subtracted. The default is True.
        standardize : bool, optional
            If True, every column will be standardized. The default is True.
        verbose : bool, optional
            If True, the number of iterations for the convergence of 
            the algorithm will be logged to console. The default is True.
        max_iterations : int, optional
            Maximum number of iterations before stopping. The default is 10000.

        Returns
        -------
        Scores, loadings, residuals, R square, Explained variance and eigenvalues of the generated
        PCA model.

        '''
        
        if ncomps==None:
            self._ncomps = self._nvars
        else:
            self._ncomps = ncomps

        self._scores, self._loadings, self._residuals, self._rsquare, self._explained_variance, self._eigenvals = \
            nipals(self._X, ncomps=self._ncomps, 
                   threshold=threshold, 
                   demean=demean, 
                   standardize=standardize, 
                   verbose=verbose, 
                   max_iterations=max_iterations)

    def hotelling_T2(self, alpha, plot=True):
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
        plot : bool, optional
            If True, a plot of the Hotelling's T2 statistic and the control limit will be displayed. The default is True.

        Returns
        -------
        Hotelling's T2 statistic for every observation.

        '''

        dfn = self._ncomps/2
        dfd = (self._nobs-self._ncomps-1)/2
        const = ((self._nobs-1)**2)/self._nobs

        # Cálculo de la T2 de Hotelling
        T_2 = []
        for i in range(self._scores.shape[0]):

            z = self._scores[i, :]
            t2 = np.sum((z**2)/self._eigenvals)
            T_2.append(t2)

        self._hotelling = np.array(T_2)
        self._hotelling_limit = beta.ppf(alpha, dfn, dfd)*const
        
        if plot:
            
            fig, ax = plt.subplots()
            ax.plot(self._hotelling)
            ax.set_title(f"Hotelling's T2 with alpha={alpha*100:.2f}%")
            ax.set_xlabel("Observations")
            ax.set_ylabel("Hotelling's T2 value")

            ax.axhline(self._hotelling_limit, color='red')
            plt.show()
    
    def spe(self, alpha, plot=True):
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
        
        SPE = []
        
        for i in range(self._residuals.shape[0]):
            e = self._residuals[i, :]
            SPE.append(np.transpose(e).dot(e))

        b, nu = np.mean(SPE), np.var(SPE)
        
        df = (2*b**2)/nu
        const = nu/(2*b)

        self._spe = SPE
        self._spe_limit = chi2.ppf(alpha, df)*const
        
        if plot:

            fig, ax = plt.subplots()
            ax.plot(self._spe)
            ax.set_title(f"SPE with alpha={alpha*100:.2f}%")
            ax.set_xlabel("Observations")
            ax.set_ylabel("SPE value")

            ax.axhline(self._spe_limit, color='red')
            plt.show()
        
        
    def spe_contribution(self):
        '''
        Contribution of each measured kth variable to the SPE statistic for every observation.

        Returns
        -------
        Contributions to the SPE statistic.

        '''
        
        self._spe_contribution = self._residuals**2
        
    def T2_contribution(self):
        '''

        The diagnosis procedure is carried out in two steps: (i) a bar plot of the normalized scores for that observation
        (t_new,a/lambda_a)**2 is calculated and the ath score with the highest normalized value is selected; (ii) the 
        contribution of each measured kth variable to this ath score at this new abnormal observation is given by 
                Cont(t_new,a ; x_new,k) = p_ak * x_new,k
        where p_ak is the loading of the kth variable at component a
        
        Variables with high contributions but with the same sign as the score should be investigated (contributions of
        the opposite sign will only make the score smaller). 
        

        Returns
        -------
        Contributions of the variables to the T2 statistic for every observation.

        '''
    
        T2_1 = (self._scores/self._eigenvals[None, :])**2
        components = np.argmax(T2_1, axis=1)

        T2_2 = np.empty(shape=(self._X.shape[0], self._X.shape[1]))
        
        mean = np.mean(self._X, axis=0)
        std = np.std(self._X, axis=0)
        for obs, a in enumerate(components):
            p_a = self._loadings[a, :]
            x_i = (self._X[obs, :] - mean)/std
            cont = p_a*x_i
            T2_2[obs, :] = cont
            
        self._T2_contribution = T2_2


    def score_plot(self, comp_x, comp_y):
        '''
        Plot to compare the scores of 2 given components

        Parameters
        ----------
        comp_x : int
            First component to be plotted.
        comp_y : int
            Second component to be plotted.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        Score plot.

        '''
        if (comp_x<=0) or (comp_y<=0):
            raise ValueError("Las componentes no pueden ser menores o iguales que 0")
    
        var1, var2 = self._explained_variance[comp_x - 1], self._explained_variance[comp_y - 1]
        total_variance = var1+var2
        
        Comp_1 = self._scores[:, comp_x-1]
        Comp_2 = self._scores[:, comp_y-1]
    
        fig, ax = plt.subplots()
        sns.scatterplot(x=Comp_1, y=Comp_2, alpha=1)
        ax.set_title(f'Score plot for components {comp_x} and {comp_y}. Explained variance: {total_variance*100:.2f}%')
        ax.set_xlabel(f"Component {comp_x} ({var1 * 100:.2f}%)")
        ax.set_ylabel(f"Component {comp_y} ({var2 * 100:.2f}%)")
        ax.axvline(x=0, color='black')
        ax.axhline(y=0, color='black')

    def loadings_plot(self, comp):
        '''
        Plots the loadings of the variables in each component

        Parameters
        ----------
        comp : int
            Principal component to be plotted.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        Loadings plot.

        '''
        if (comp<=0):
            raise ValueError("La componente no puede ser menor o igual que 0")
    
        fig, ax = plt.subplots()
        ax.bar(x=range(self._loadings.shape[1]), height=self._loadings[comp-1, :])
        ax.set_title(f'Loadings plot for component {comp}')
        plt.show()

    def compare_loadings(self, comp_x, comp_y):
        '''
        Scatter plot to compare the loadings of two principal components

        Parameters
        ----------
        comp_x : int
            First component to be plotted.
        comp_y : int
            Second component to be plotted.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if (comp_x<=0) or (comp_y<=0):
            raise ValueError("Las componentes no pueden ser menores o iguales que 0")
       
        Comp_1 = self._loadings[:, comp_x-1]
        Comp_2 = self._loadings[:, comp_y-1]
    
        fig, ax = plt.subplots()
        sns.scatterplot(x=Comp_1, y=Comp_2, alpha=1)
        var1, var2 = self._explained_variance[comp_x - 1], self._explained_variance[comp_y - 1]
        total_variance = var1+var2
        
        
        ax.set_title(f'Loadings plot for components {comp_x} and {comp_y}. Explained variance: {total_variance*100:.2f}%')
        ax.set_xlabel(f"Component {comp_x} ({var1 * 100:.2f}%)")
        ax.set_ylabel(f"Component {comp_y} ({var2 * 100:.2f}%)")
        ax.axvline(x=0, color='black')
        ax.axhline(y=0, color='black')
        plt.show()

