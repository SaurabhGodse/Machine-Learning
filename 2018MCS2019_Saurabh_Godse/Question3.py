import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv as inv

if(len(sys.argv) == 2):
    file1 = sys.argv[1]
    file2 = sys.argv[2]
else:
    file1 = "logisticX.csv"
    file2 = "logisticY.csv"
    
X = np.loadtxt(file1, delimiter = ",")
Y = np.loadtxt(file2, delimiter = ",")
plt.scatter(X[:,0], X[:,1], c=Y)

X = np.array([[1,x[0], x[1]] for x in X])
(M, N) = np.shape(X)
  

def g(Theta):
    return 1 / (1 + np.exp(-1 * (X @ Theta)))

def Delta(pi):
    return -1 * (X.T @ (pi - Y))

def Hessian(pi):
    temp = []
    for r in pi:
        temp.append(r * (1 - r))
    return -1 * (X.T @ np.diag(temp) @ X)

def Logistic():
    PrevTheta = np.zeros(N)
    IsConverged = False
    iterations = 0
    while(not IsConverged):
        pi = g(PrevTheta)
        delta = Delta(pi)
        Hess = Hessian(pi)
        CurrentTheta = PrevTheta - inv(Hess) @ delta
        Difference = abs(CurrentTheta - PrevTheta)
        (flag, ) = np.shape(Difference)
        j = 0
        
        for i in Difference:
            if(i < 10 ** -9):
                j += 1
        if(j == flag):
            print("Converged Successfully")
            IsConverged = True
        iterations += 1
        if(iterations == 10000):
            print("Terminated")
            break
        PrevTheta = CurrentTheta
    print("Current Theta : ", CurrentTheta)
    print("Iterations : ", iterations)
    x = np.linspace(1,9,100)
    y = -1 * ((CurrentTheta[1] * x + CurrentTheta[0]) / CurrentTheta[2])
    plt.plot(x, y, 'g-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Logistic Regression")
    plt.savefig("Q3_b.png")

    plt.show()

Logistic()