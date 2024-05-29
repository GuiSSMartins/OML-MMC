import numpy as np
import math
import matplotlib.pyplot as plt

def calculo_accuracy(X, Y, w):
    #Calculando a acurácia
    correct_predictions = 0
    previsao = []
    for i in range(len(X)):
        prediction = sigmoid(np.dot(w, X[i]))
        #print("Preveu ",prediction)
        if sinal(prediction) == Y[i]:
            correct_predictions += 1
        previsao.append(sinal(prediction))
    accuracy = correct_predictions / len(X)
    return previsao

def calcular_erro(Y, p_chapeu):
    N = len(p_chapeu)
    erro = 0
    for n in range(N):
        erro += -(Y[n] * np.log(p_chapeu[n]) + (1 - Y[n]) * np.log(1 - p_chapeu[n]))
    return erro / N


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500))))
def sinal(z):
    return 1 if z > 0.5 else 0

def calculo_S(p_chapeu, Y, X):
    N = len(p_chapeu)
    S = 0
    for n in range(N):
        S += (p_chapeu[n] - Y[n]) * X[n]
    return S / N

def calcula_p_chapeu(w, X):
    p_chapeu =[]
    for n in range(len(X)):
        p_chapeu.append(sigmoid(np.dot(w,X[n])))
    return p_chapeu      
        
def batche(X, Y, percent_to_keep):
    num_to_keep = math.ceil(len(X) * percent_to_keep)
    if num_to_keep >= len(X):
        return X.copy(), Y.copy()
    indices_to_keep = np.random.choice(len(X), num_to_keep, replace=False)
    X_batch = X[indices_to_keep]
    Y_batch = Y[indices_to_keep]
    return X_batch, Y_batch       

# Algoritmo conforme o pseudocódigo
def MGmB(X, Y, w, eta,batch_size,epochs):
    t=0
    E=[]
    batch_size = batch_size/len(X)
    for epoca in range(epochs):
        X_batch, Y_batch = batche(X,Y,batch_size)
        p_chapeu = calcula_p_chapeu(w, X_batch)
        S = calculo_S(p_chapeu, Y_batch, X_batch)
        w = w - eta * S
        #print(calculo_accuracy(X, Y, w) )
        E.append(calcular_erro(Y_batch, p_chapeu))
        #print("Usando : ", len(X_batch)," Elementos")
    return w, E
def MGE(X, Y, w, eta,epochs):
    t=0
    E=[]
    batch_size = 1/len(X)
    for epoca in range(epochs):
        X_batch, Y_batch = batche(X,Y,batch_size)
        p_chapeu = calcula_p_chapeu(w, X_batch)
        S = calculo_S(p_chapeu, Y_batch, X_batch)
        w = w - eta * S
        #print(calculo_accuracy(X, Y, w) )
        E.append(calcular_erro(Y_batch, p_chapeu))
        #print("Usando : ", len(X_batch)," Elementos")
    return w, E