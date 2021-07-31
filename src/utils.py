#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:40:07 2021

@author: fidae
"""

import numpy as np
from numpy import linalg as LA


def nipals(X, ncomps, threshold=1e-5, demean=True, standardize=True, verbose=True, max_iterations=10000):
  
    X_pca = X.copy()

    if demean:
        mean = np.mean(X, axis=0)
        X_pca = X_pca-mean[None, :]
        
    if standardize:
        std = np.std(X, axis=0)
        X_pca = X_pca/std[None, :]


    sct = np.sum(X_pca**2)
    
    r2 = []
    T = np.zeros(shape=(ncomps, X_pca.shape[0]))
    P_t = np.zeros(shape = (ncomps, X_pca.shape[1]))
    vals= np.zeros(ncomps)

    for i in range(ncomps):
        # We get the column with the maximum variance in the matrix
        var = np.var(X_pca, axis=0)
        pos = np.where(max(var))[0]
        
        # That column will be the one we will start with
        t = np.array(X_pca[:,pos])
        t.shape=(X_pca.shape[0], 1) #Esto sirve para obligar a que t sea un vector columna
    
        cont=0
        comprobacion = 1
        # while conv <X_pca.shape[0] and cont<10000:
        while comprobacion>threshold and cont<max_iterations:
            
            #Definimos un vector llamado t_previo, que es con el que vamos a empezar el algoritmo
            t_previo = t
            p_t = (np.transpose(t_previo).dot(X_pca))/(np.transpose(t_previo).dot(t_previo))
            p_t = p_t/LA.norm(p_t)
            
            t=X_pca.dot(np.transpose(p_t))
            
            #Comparamos el t calcular con el t_previo, de manera que lo que buscamos es que la diferencia sea menor
            #que el criterio de parada establecido. Para ello, hacemos una prueba lógica y sumamos todos los valores
            #donde sea verdad. Si es verdad en todos, el algoritmo ha convergido
            
            t_sum = np.sum(t**2)
            t_previo_sum = np.sum(t_previo**2)
            
            comprobacion = np.abs(np.sqrt(t_sum-t_previo_sum))
            
            cont+=1

            
        #Calculamos la matriz de residuos y se la asignamos a X para calcular la siguiente componente
        E = X_pca-t.dot(p_t)
        r2.append(1-np.sum(E**2)/sct)
        X_pca=E
        
        #Asignamos los vectores t y p a su posición en las matrices de scores y loadings
        vals[i] = np.var(t)
        
        T[i]=t.reshape((X.shape[0]))
        P_t[i]=p_t
        
    if verbose:
        print(f"Algorithm converged in {cont} iterations")
    T = np.transpose(T)
    
    
    return T, P_t, r2, vals

# import pandas as pd
# from sklearn.impute import KNNImputer
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# df = pd.read_csv("../data/water_potability.csv")

# knn_imp = KNNImputer()
# X = knn_imp.fit_transform(df)


# T, P_t, r2, vals = nipals(X, 5)

# ss = StandardScaler()
# X_pr = ss.fit_transform(X)
# pca = PCA(n_components=5)
# pca.fit(X_pr)
# pca.explained_variance_ratio_
