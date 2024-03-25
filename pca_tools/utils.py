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


    tss = np.sum(X_pca**2)
    
    r2 = []
    explained_variance = []
    T = np.zeros(shape=(ncomps, X_pca.shape[0]))
    P_t = np.zeros(shape = (ncomps, X_pca.shape[1]))
    eigenvalues= np.zeros(ncomps)

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
        r2.append(1-np.sum(E**2)/tss)
        explained_variance.append(r2[i] - r2[i-1]) if i!=0 else explained_variance.append(r2[i])
        X_pca = E
        
        #Asignamos los vectores t y p a su posición en las matrices de scores y loadings
        eigenvalues[i] = np.var(t)
        
        T[i]=t.reshape((X.shape[0]))
        P_t[i]=p_t
        
    if verbose:
        print(f"Algorithm converged in {cont} iterations")
    T = np.transpose(T)
    
    
    return T, P_t, E, r2, explained_variance, eigenvalues

# def optimize_T2(X, ncomps, alpha, df_metadatos, threshold=3):
    
#     X_opt = X.copy()
#     tam = X.shape[0]
#     fuera_control = tam
#     pos_elim = np.array([])
#     df_meta = df_metadatos.copy()

#     umbral = int(tam*(1-alpha))
    
#     while fuera_control > threshold*umbral:
#         pca = miPCA(ncomps)
#         pca.calcular(X_opt)
        
#         T = pca._scores
#         lam_a = pca._eigenvals
#         obs = X_opt.shape[0]
        
#         dfn = ncomps/2
#         dfd = (obs-ncomps-1)/2
#         const = ((obs-1)**2)/obs
    
#         T_2 = []
#         for i in range(T.shape[0]):
#             t2 = 0
#             z = T[i, :]
#             t2 = np.sum((z**2)/lam_a)
#             T_2.append(t2)
            
#         T_2 = np.array(T_2)
#         t_lim = (beta.ppf(alpha, dfn, dfd))*const
        
#         # pos_max = np.argmax(T_2)
        
#         fuera_control = np.sum(T_2>t_lim)
#         umbral = int(tam*(1-alpha))
#         eliminar = int(alpha*(fuera_control-umbral))
        
#         pos_max = T_2.argsort()[-eliminar:][::-1]
#         pos_elim = np.append(pos_elim, pos_max)
#         X_opt = np.delete(X_opt, pos_max, 0)
#         df_meta.drop(pos_max, axis=0, inplace=True)
#         df_meta = df_meta.reset_index(drop=True)
        
        
#         tam = X_opt.shape[0]
        
#         print("#######################################")
#         print(len(pos_elim), " Elementos eliminados")
#         print("Fuera de control:", fuera_control)
#         print("Umbral: ", threshold*umbral)
#         print("Límite de control: ", t_lim)

#     model = miPCA(ncomps, autoesc=True)
#     model.calcular(X_opt)
#     model.hotelling(alpha)
    
#     t_lim = model._hotelling_limit
#     T2 = model._hotelling
#     return X_opt, model, T2, t_lim, pos_elim, df_meta

# def optimize_SPE(X, ncomps, alpha, df_metadatos, threshold=1.5):
    
#     X_opt = X.copy()
#     tam = X.shape[0]
#     fuera_control = tam
#     pos_elim = []
#     df_meta = df_metadatos.copy()
 

#     umbral = int(tam*(1-alpha))
    
#     while fuera_control>threshold*umbral:
#         pca = miPCA(ncomps)
#         pca.calcular(X_opt)
#         pca.spe(alpha)
#         spe = pca._spe
    
#         b = np.mean(spe)
#         nu = np.var(spe)
        
#         df = (2*b**2)/nu
#         const = nu/(2*b)
        
#         spe_lim = chi2.ppf(alpha, df)*const
        
#         pos_max = np.argmax(spe)
#         pos_elim.append(pos_max)
#         X_opt = np.delete(X_opt, pos_max, 0)
#         df_meta.drop(pos_max, axis=0, inplace=True)
#         df_meta = df_meta.reset_index(drop=True)
        
#         fuera_control = np.sum(spe>spe_lim)
#         tam = X_opt.shape[0]
#         umbral = int(tam*(1-alpha))
#         print("#######################################")
#         print(len(pos_elim), " Elementos eliminados")
#         print("Fuera de control:", fuera_control)
#         print("Umbral: ", threshold*umbral)
#         print("Límite de control: ", spe_lim)