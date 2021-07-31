# import pandas as pd
# import numpy as np
# # from statsmodels.multivariate.pca import PCA
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import beta, chi2
# from sklearn.cluster import KMeans
# from sklearn.neighbors import DistanceMetric

import numpy as np
import pandas as pd
from utils import nipals
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, chi2
# from sklearn.cluster import DBSCAN, OPTICS

np.seterr(divide='ignore', invalid='ignore')

class PCA():
    def __init__(self, X):
        
        self._nobs = X.shape[0]
        self._nvars = X.shape[1]
        self._X = X

    def fit(self, ncomps, threshold=1e-12, demean=True, standardize=True, verbose=True, max_iterations=10000):
        
        self._ncomps = ncomps

        self._scores, self._loadings, self._residuals, self._rsquare, self._explained_variance, self._eigenvals = \
            nipals(self._X, ncomps=ncomps, 
                   threshold=threshold, 
                   demean=demean, 
                   standardize=standardize, 
                   verbose=verbose, 
                   max_iterations=max_iterations)

    def hotelling_T2(self, alpha, plot=True):
        '''
        T2 de Hotelling según la fórmula T2 = suma(t_a**2/lambda_a), siendo lambda_a el autovalor de la columna
        a de la matriz de scores, y t_a el score de la observación i. El límite de control se obtiene 
        calculando la distribución Beta con A/2 y (m-A-1)/2 grados de libertad, siendo m el número de observaciones
        '''

        dfn = self._ncomps/2
        dfd = (self._nobs-self._ncomps-1)/2
        const = ((self._nobs-1)**2)/self._nobs

        # Cálculo de la T2 de Hotelling
        T_2 = []
        for i in range(self._scores.shape[0]):
            t2 = 0
            z = self._scores[i, :]
            t2 = np.sum((z**2)/self._eigenvals)
            T_2.append(t2)

        self._hotelling = np.array(T_2)
        self._hotelling_limit = beta.ppf(alpha, dfn, dfd)*const
        
        if plot:
            plt.plot(self._hotelling)
            plt.axhline()

            fig, ax = plt.subplots()
            ax.plot(self._hotelling)
            ax.set_title(f"Hotelling's T2 with alpha={alpha*100:.2f}%")
            ax.set_xlabel("Observations")
            ax.set_ylabel("Hotelling's T2 value")

            ax.axhline(self._hotelling_limit, color='red')
            plt.show()
    
    def spe(self, alpha, plot=True):
        
        spe = []
        
        for i in range(self._residuals.shape[0]):
            e = self._residuals[i, :]
            val = np.transpose(e).dot(e)
            spe.append(val)

        b, nu = np.mean(spe), np.var(spe)
        
        df = (2*b**2)/nu
        const = nu/(2*b)

        self._spe = spe
        self._spe_limit = chi2.ppf(alpha, df)*const
        
        if plot:
            plt.plot(self._hotelling)
            plt.axhline()

            fig, ax = plt.subplots()
            ax.plot(self._spe)
            ax.set_title(f"SPE with alpha={alpha*100:.2f}%")
            ax.set_xlabel("Observations")
            ax.set_ylabel("SPE value")

            ax.axhline(self._spe_limit, color='red')
            plt.show()
        
        
    def spe_contribution(self):
        
        self._spe_contribution = self._residuals**2
        
    def T2_contribution(self):
        
        '''
        El análisis de la contribución de la T2 se hace en dos fases. En la primera
        se obtiene la relación (t/lambda) 2, y se extrae la componente que devuelva el mayor valor.
        A continuación, se calcula la relación p_ak*x_ik, donde a es la componente seleccionada, k son las variables
        e i es la observación
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


    def score_plot(self, comp1, comp2, hue=None, legend=True):
        if (comp1<=0) or (comp2<=0):
            raise ValueError("Las componentes no pueden ser menores o iguales que 0")
    
        var1, var2 = self._explained_variance[comp1 - 1], self._explained_variance[comp2 - 1]
        
        Comp_1 = self._scores[:, comp1-1]
        Comp_2 = self._scores[:, comp2-1]
    
        fig, ax = plt.subplots()
        sns.scatterplot(x=Comp_1, y=Comp_2, hue=hue, alpha=1, legend=legend)
        ax.set_title(f'Gráfico de scores para las componentes {comp1} y {comp2}')
        ax.set_xlabel(f"Componente {comp1} ({var1 * 100:.2f}%)")
        ax.set_ylabel(f"Componente {comp2} ({var2 * 100:.2f}%)")
        ax.axvline(x=0)
        ax.axhline(y=0)

    def loadings_plot(self, comp):
        if (comp<=0):
            raise ValueError("La componente no puede ser menor o igual que 0")
    
        fig, ax = plt.subplots()
        ax.bar(x=range(self._loadings.shape[0]), height=self._loadings[:, comp-1])
        ax.set_title(f'Gráfico de loadings para la componente {comp}')
        plt.show()

    def compare_loadings(self, comp1, comp2, hue=None, legend=True):
        if (comp1<=0) or (comp2<=0):
            raise ValueError("Las componentes no pueden ser menores o iguales que 0")
       
        Comp_1 = self._loadings[:, comp1-1]
        Comp_2 = self._loadings[:, comp2-1]
    
        fig, ax = plt.subplots()
        sns.scatterplot(x=Comp_1, y=Comp_2, hue=hue, alpha=1, legend=legend)
        var1, var2 = self._explained_variance[comp1 - 1], self._explained_variance[comp2 - 1]
        ax.set_xlabel(f"Componente {comp1} ({var1 * 100:.2f}%)")
        ax.set_ylabel(f"Componente {comp2} ({var2 * 100:.2f}%)")
        ax.axvline(x=0)
        ax.axhline(y=0)
        plt.show()
        


