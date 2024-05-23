import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sinal(z):
    return 1 if z > 0.5 else 0
def CLogDKPd_MGE(X, Y, alpha, eta,epochs,d):
    t=0
    E=[]
    N=len(X)
    A=np.dot(X,X.T)
    for epoca in range(epochs):
        ordem =np.random.randint(0, N-1)
        ordem =2
        x_n = X[ordem]
        p_chapeu = sigmoid(sum(alpha * A[ordem]**d ))
        S = ((p_chapeu-Y[ordem])*A[ordem])
        alpha = alpha - eta*S
        print(alpha)