import pandas as pd
import numpy as np
# from statsmodels.multivariate.pca import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric


from sklearn.cluster import DBSCAN, OPTICS

from nipals import PCA

np.seterr(divide='ignore', invalid='ignore')

class miPCA(PCA):
    def __init__(self, ncomps, autoesc=True, verbose=False):
        self._ncomps = ncomps
        self._verbose = verbose
        self._autoesc = autoesc
        self._residuals = None
        self._loadings = None
        self._scores = None
        self._tagged_scores = None
        self._tagged_loadings = None
        self._plot = None
        self._rsquare = None
        self._explained_variance = None
        self._eigenvals = None
        self._hotelling = None
        self._hotelling_limit = None
        self._nvars = None
        self._X = None
        self._spe_contrib = None
        self._T2_contrib = None
        self._clusters_T_inertia = None
        self._clusters_P_inertia = None

    def calcular(self, X):
        
        ncomps = self._ncomps
        pca = PCA()
        pca.nipals(X, ncomps=ncomps, tol=1e-2, autoesc=self._autoesc, verbose=self._verbose)

        x = pca._rsquared_acc

        self._explained_variance = np.array([x[0]]+[x[i]-x[i-1] for i in range(1, len(x))])
        self._nvars = X.shape[1]
        self._residuals = pca._residuals
        self._loadings = pca._loadings
        self._scores = pca._scores
        self._rsquare = pca._rsquared_acc
        self._eigenvals = pca._eigenvals
        self._X = X
    
    def cluster(self, n_clusters=5):
        
        T = self._scores
        P = np.transpose(self._loadings)

        clust_T = KMeans(n_clusters=n_clusters)
        clust_T.fit(T)
        
        clust_P = KMeans(n_clusters=n_clusters)
        clust_P.fit(P)
        
        self._clusters_scores = clust_T.labels_
        self._clusters_loadings = clust_P.labels_
        self._clusters_T_inertia = clust_T.inertia_
        self._clusters_P_inertia = clust_P.inertia_
    def hotelling(self, alpha):
        '''
        T2 de Hotelling según la fórmula T2 = suma(t_a**2/lambda_a), siendo lambda_a el autovalor de la columna
        a de la matriz de scores, y t_a el score de la observación i. El límite de control se obtiene 
        calculando la distribución Beta con A/2 y (m-A-1)/2 grados de libertad, siendo m el número de observaciones
        '''
        T = self._scores
        lam_a = self._eigenvals
        obs = self._scores.shape[0]
        A = self._ncomps

        dfn = A/2
        dfd = (obs-A-1)/2
        const = ((obs-1)**2)/obs

        # Cálculo de la T2 de Hotelling
        T_2 = []
        for i in range(T.shape[0]):
            t2 = 0
            z = T[i, :]
            t2 = np.sum((z**2)/lam_a)
            T_2.append(t2)

        lim_control = beta.ppf(alpha, dfn, dfd)*const

        self._hotelling = np.array(T_2)
        self._hotelling_limit = lim_control
        
    def contributions(self):
        
        P_t = self._loadings
        E = self._residuals
        X = self._X
        T = self._scores
        lam = self._eigenvals


        spe_contribution = E**2

        '''
        El análisis de la contribución de la T2 se hace en dos fases. En la primera
        se obtiene la relación (t/lambda) 2, y se extrae la componente que devuelva el mayor valor.
        A continuación, se calcula la relación p_ak*x_ik, donde a es la componente seleccionada, k son las variables
        e i es la observación
        '''
        T2_1 = (T/lam[None, :])**2
        components = np.argmax(T2_1, axis=1)

        T2_2 = np.empty(shape=(X.shape[0], X.shape[1]))
        
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        for obs, a in enumerate(components):
            p_a = P_t[a, :]
            x_i = (X[obs, :] - mean)/std
            cont = p_a*x_i
            T2_2[obs, :] = cont

        self._spe_contrib = spe_contribution
        self._T2_contrib = T2_2

    def spe(self, alpha):
        E = self._residuals
        
        spe = []
        
        for i in range(E.shape[0]):
            e = E[i, :]
            val = np.transpose(e).dot(e)
            spe.append(val)


        b, nu = np.mean(spe), np.var(spe)
        
        df = (2*b**2)/nu
        const = nu/(2*b)
        
        spe_lim = chi2.ppf(alpha, df)*const

            
        self._spe = spe
        self._spe_limit = spe_lim
            
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
        