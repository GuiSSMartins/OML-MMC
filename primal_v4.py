import numpy as np


# Perceptron Primal V4

# T = nº epochs
def primal_v4(x_til,y,w_0,T,miu):
    t=0
    w_til=[w_0]
    E=[]
    count=[] # misclassified labels
    N = len(x_til)
    for tau in range(T):
        count[tau] = 0
        y_chapeu = []
        for n in range(N):
            y_chapeu[n] = np.sign(np.dot(w_til[t],x_til[n]))
            if y_chapeu[n] == 0:
                y_chapeu[n] = -1
            if y_chapeu[n] != y[n]:
                w_til.append(w_til[-1] + (miu/2)*(y_chapeu[n] - y[n])*x_til[n])
                count[tau] = count[tau] + 1
            t = t + 1
            custo_t =[(miu/2)*abs(y[p]-y_chapeu[p] for p in range(N))]  
            E.append((1/N)*sum(custo_t))
        if count[tau] == 0:
            return w_til[-1]
    return w_til[-1]