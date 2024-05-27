import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

#def sigmoid(z):
    #return 1 / (1 + np.exp(-z))
    

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def threshold(z):
    return 1 if z > 0.5 else 0
def batche(lista, n):
    # Seleciona aleatoriamente n índices únicos da lista
    indices_aleatorios = random.sample(range(len(lista)), n)
    return indices_aleatorios



def calcular_erro(alpha, X, Y, A):
    erro = 0
    N = len(X)
    epsilon = 1e-12
    for n in range(N):
        p_chapeu = sigmoid(np.sum(alpha * A[n]))
        erro -= (Y[n] * np.log(p_chapeu + epsilon) + (1 - Y[n]) * np.log(1 - p_chapeu + epsilon))
    erro = max(erro / N, epsilon)
    return erro



#def calcular_erro(alpha, X, Y, A):
#   erro = 0
#    N = len(X)
#    for n in range(N):
#        p_chapeu = sigmoid(sum(alpha * (A[n]) ))
#        if(1 - p_chapeu)<np.e**-12:
#            erro -= np.e**-12
#        else:
#            erro -= (Y[n] * np.log(p_chapeu) + (1 - Y[n]) * np.log(1 - p_chapeu))
#    erro = erro/N
#    return erro if erro > np.e**-12 else np.e**-12

def CLogDKPd_MGmB(X, Y, alpha, eta,epochs,d,batch_size):
    t=20
    E=[]
    N=len(X)
    A=np.dot(X,X.T)
    A=A**d
    count = 0
    for epoca in range(epochs):
        S=[]
        ordem =batche(X, batch_size)
        for i in ordem:
            p_chapeu=sigmoid(sum(alpha * (A[i])))
            S.append((p_chapeu-Y[i]) *(A[i]))
        S=np.array(np.sum(S,axis=0))
        alpha = alpha - eta * S
        if t ==20:
            E.append(calcular_erro(alpha, X, Y, A))
            t = 0
        t +=1
        
    return alpha, E
def CLogDKPd_MGE_Ordenado(X, Y, alpha, eta,epochs,d):
    t=0
    E=[]
    N=len(X)
    A=np.dot(X,X.T)
    A=A**d
    for epoca in range(epochs):
        if t>= N:
            t=0
        ordem = t
        p_chapeu = sigmoid(sum(alpha * (A[ordem]) ))
        S = (p_chapeu-Y[ordem]) *(A[ordem])
        alpha = alpha - eta * S
        E.append(calcular_erro(alpha, X, Y, A))
        t +=1
    return alpha, E
def CLogDKPd_MGE(X, Y, alpha, eta,epochs,d):
    t=0
    E=[]
    N=len(X)
    A=np.dot(X,X.T)
    A=A**d
    for epoca in range(epochs):
        ordem =np.random.randint(0, N)
        p_chapeu = sigmoid(sum(alpha * (A[ordem]) ))
        S = (p_chapeu-Y[ordem]) *(A[ordem])
        alpha = alpha - eta * S
        E.append(calcular_erro(alpha, X, Y, A))
    return alpha, E