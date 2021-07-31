import pandas as pd
import numpy as np
# from statsmodels.multivariate.pca import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric

from utils import nipals

from sklearn.cluster import DBSCAN, OPTICS

np.seterr(divide='ignore', invalid='ignore')

class PCA():
    def __init__(self, X):
        
        self._nobs = X.shape[0]
        self._nvars = X.shape[1]
        self._X = X

    def fit(self, X, ncomps, threshold=1e-12, demean=True, standardize=True, verbose=True, max_iterations=10000):
        
        self._ncomps = ncomps

        self._scores, self._loadings, self._residuals, self._rsquare, self._eigenvals = \
            nipals(X, ncomps=ncomps, 
                   threshold=threshold, 
                   demean=demean, 
                   standardize=standardize, 
                   verbose=verbose, 
                   max_iterations=max_iterations)

    
    # def cluster(self, n_clusters=5):
        
    #     T = self._scores
    #     P = np.transpose(self._loadings)

    #     clust_T = KMeans(n_clusters=n_clusters)
    #     clust_T.fit(T)
        
    #     clust_P = KMeans(n_clusters=n_clusters)
    #     clust_P.fit(P)
        
    #     self._clusters_scores = clust_T.labels_
    #     self._clusters_loadings = clust_P.labels_
    #     self._clusters_T_inertia = clust_T.inertia_
    #     self._clusters_P_inertia = clust_P.inertia_

    def hotelling_T2(self, alpha):
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
        
    def spe(self, alpha):
        
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

            
    def etiquetas(self, df_metadata, variables):

        if df_metadata.shape[0] != self._scores.shape[0]:
            raise ValueError("Las dimensiones de los metadatos y los scores no son iguales")

        T = self._scores
        P_t = self._loadings

        df_T = pd.DataFrame(T)
        df_P = pd.DataFrame(np.transpose(P_t))
        df_P.columns = [f"Comp_{i}" for i in range(1, self._ncomps+1)]

        df_T.columns = [f"Comp_{i}" for i in range(1, self._ncomps+1)]
        if self._clusters_scores is not None:
            df_T['Clusters'] = self._clusters_scores
            df_P['Clusters'] = self._clusters_loadings 

        df_vars = pd.DataFrame(variables)
        df_scores = pd.concat([df_metadata, df_T], axis=1)
        df_loadings = pd.concat([df_vars, df_P], axis=1)

        self._tagged_scores = df_scores
        self._tagged_loadings = df_loadings

def score_plot(pca_object, comp1, comp2, hue=None, legend=True):
    if (comp1<=0) or (comp2<=0):
        raise ValueError("Las componentes no pueden ser menores o iguales que 0")

    df = pca_object._tagged_scores
    var1, var2 = pca_object._explained_variance[[comp1 - 1, comp2 - 1]]

    fig, ax = plt.subplots()
    sns.scatterplot(x=f"Comp_{comp1}", y=f"Comp_{comp2}", data=df, hue=hue, alpha=1, legend=legend)
    ax.set_title(f'Gráfico de scores para las componentes {comp1} y {comp2}')
    ax.set_xlabel(f"Componente {comp1} ({var1 * 100:.2f}%)")
    ax.set_ylabel(f"Componente {comp2} ({var2 * 100:.2f}%)")
    ax.axvline(x=0)
    ax.axhline(y=0)

def loadings_plot(pca_object, comp):
    if (comp<=0):
        raise ValueError("La componente no puede ser menor o igual que 0")

    df_loadings = pca_object._tagged_loadings

    fig, ax = plt.subplots()
    ax.bar(x=range(df_loadings.shape[0]), height=df_loadings.iloc[:, comp])
    ax.set_title(f'Gráfico de loadings para la componente {comp}')
    plt.show()

def compare_loadings(pca_object, comp1, comp2, hue, legend):
    if (comp1<=0) or (comp2<=0):
        raise ValueError("Las componentes no pueden ser menores o iguales que 0")

    df = pca_object._tagged_loadings

    fig, ax = plt.subplots()
    sns.scatterplot(x=f"Comp_{comp1}", y=f"Comp_{comp2}", data=df, hue=hue, alpha=1, legend=legend)
    var1, var2 = pca_object._explained_variance[[comp1 - 1, comp2 - 1]]
    ax.set_xlabel(f"Componente {comp1} ({var1 * 100:.2f}%)")
    ax.set_ylabel(f"Componente {comp2} ({var2 * 100:.2f}%)")
    ax.axvline(x=0)
    ax.axhline(y=0)
    plt.show()

def optimize_T2(X, ncomps, alpha, df_metadatos, threshold=3):
    
    X_opt = X.copy()
    tam = X.shape[0]
    fuera_control = tam
    pos_elim = np.array([])
    df_meta = df_metadatos.copy()

    umbral = int(tam*(1-alpha))
    
    while fuera_control > threshold*umbral:
        pca = miPCA(ncomps)
        pca.calcular(X_opt)
        
        T = pca._scores
        lam_a = pca._eigenvals
        obs = X_opt.shape[0]
        
        dfn = ncomps/2
        dfd = (obs-ncomps-1)/2
        const = ((obs-1)**2)/obs
    
        T_2 = []
        for i in range(T.shape[0]):
            t2 = 0
            z = T[i, :]
            t2 = np.sum((z**2)/lam_a)
            T_2.append(t2)
            
        T_2 = np.array(T_2)
        t_lim = (beta.ppf(alpha, dfn, dfd))*const
        
        # pos_max = np.argmax(T_2)
        
        fuera_control = np.sum(T_2>t_lim)
        umbral = int(tam*(1-alpha))
        eliminar = int(alpha*(fuera_control-umbral))
        
        pos_max = T_2.argsort()[-eliminar:][::-1]
        pos_elim = np.append(pos_elim, pos_max)
        X_opt = np.delete(X_opt, pos_max, 0)
        df_meta.drop(pos_max, axis=0, inplace=True)
        df_meta = df_meta.reset_index(drop=True)
        
        
        tam = X_opt.shape[0]
        
        print("#######################################")
        print(len(pos_elim), " Elementos eliminados")
        print("Fuera de control:", fuera_control)
        print("Umbral: ", threshold*umbral)
        print("Límite de control: ", t_lim)

    model = miPCA(ncomps, autoesc=True)
    model.calcular(X_opt)
    model.hotelling(alpha)
    
    t_lim = model._hotelling_limit
    T2 = model._hotelling
    return X_opt, model, T2, t_lim, pos_elim, df_meta

def optimize_SPE(X, ncomps, alpha, df_metadatos, threshold=1.5):
    
    X_opt = X.copy()
    tam = X.shape[0]
    fuera_control = tam
    pos_elim = []
    df_meta = df_metadatos.copy()
 

    umbral = int(tam*(1-alpha))
    
    while fuera_control>threshold*umbral:
        pca = miPCA(ncomps)
        pca.calcular(X_opt)
        pca.spe(alpha)
        spe = pca._spe
    
        b = np.mean(spe)
        nu = np.var(spe)
        
        df = (2*b**2)/nu
        const = nu/(2*b)
        
        spe_lim = chi2.ppf(alpha, df)*const
        
        pos_max = np.argmax(spe)
        pos_elim.append(pos_max)
        X_opt = np.delete(X_opt, pos_max, 0)
        df_meta.drop(pos_max, axis=0, inplace=True)
        df_meta = df_meta.reset_index(drop=True)
        
        fuera_control = np.sum(spe>spe_lim)
        tam = X_opt.shape[0]
        umbral = int(tam*(1-alpha))
        print("#######################################")
        print(len(pos_elim), " Elementos eliminados")
        print("Fuera de control:", fuera_control)
        print("Umbral: ", threshold*umbral)
        print("Límite de control: ", spe_lim)
        
    model = miPCA(ncomps, autoesc=True)
    model.calcular(X_opt)
    model.spe(alpha)
    
    spe_lim = model._spe_limit
    spe = model._spe
    return X_opt, model, spe, spe_lim, pos_elim, df_meta
        