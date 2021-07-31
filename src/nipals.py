#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:00:39 2021

@author: fidae
"""
import numpy as np
from numpy import linalg as LA

class PCA():
    def __init__(self):
        self._eigenvals = None
        self._loadings = None
        self._scores = None
        self._rsquared_acc = None
        self._residuals = None


    def nipals(self, X, ncomps, tol=1e-5, autoesc=True, verbose=True):

        nobs, nvars = X.shape

        mean, std = np.mean(X, axis=0), np.std(X, axis=0)

        if autoesc==True:
            X_pca = (X-mean[None, :])/std[None, :]
            
        else:
            X_pca = X.copy()
        
        X_sct = X_pca.copy()
        sct = np.sum(X_sct**2)
        
        r2 = []
        T = np.zeros(shape=(ncomps, X_pca.shape[0]))
        P_t = np.zeros(shape = (ncomps, X_pca.shape[1]))
        vals= np.zeros(ncomps)

        for i in range(ncomps):
            #Iniciamos t como la primera columna de X
            var = np.var(X_pca, axis=0)
            pos = np.where(max(var))[0]
            t = np.array(X_pca[:,pos])
            t.shape=(X_pca.shape[0], 1) #Esto sirve para obligar a que t sea un vector columna
            
            #Inicializamos un contador para saber en cuántas iteraciones converge el algoritmo y un criterio de parada
            cont=0
           
            comprobacion = 1
            # while conv <X_pca.shape[0] and cont<10000:
            while comprobacion>tol and cont<10000:
                
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
                
                
                #conv = np.sum(np.abs((t-t_previo))<tol)
                cont+=1
            if verbose == True:
                print("Componente ", i+1, " converge en ", cont, " iteraciones")
                
            #Calculamos la matriz de residuos y se la asignamos a X para calcular la siguiente componente
            E = X_pca-t.dot(p_t)
            r2.append(1-np.sum(E**2)/sct)
            X_pca=E
            
            #Asignamos los vectores t y p a su posición en las matrices de scores y loadings
            vals[i] = np.var(t)
            
            T[i]=t.reshape((X.shape[0]))
            P_t[i]=p_t
        T = np.transpose(T)

        self._eigenvals = vals
        self._loadings = P_t
        self._scores = T
        self._rsquared_acc = np.array(r2)
        self._residuals = E