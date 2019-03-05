import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import meshgrid

if(len(sys.argv) == 4):
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    flag = int(sys.argv[3])
else:
    file1 = "q4x.dat"
    file2 = "q4y.dat"
    flag = 1

X = np.loadtxt(file1, dtype = np.float)
Y = np.loadtxt(file2, dtype = object)

(M, N) = np.shape(X)
Y[Y == "Alaska"] = 0
Y[Y == "Canada"] = 1
Y = np.array(Y, dtype='f')
plt.scatter(X[:,0], X[:,1], c=Y)
s = np.sum(Y)
meu1 = (Y @ X)/s
meu0 = ((1-Y) @ X) / (M - s)

print("Meu0 : ", meu0)
print("Meu1 : ", meu1)

meu00 = np.outer((1 - Y), meu0)
meu11 = np.outer(Y, meu1)
meu = meu00 + meu11
phi = s / M
def GDAa():
#     Part A solution
    E = (X - meu).T @ (X - meu)
    print("Covarience when E0 = E1: ", E)
#     Part B solution
    c = ((meu0.T @ inv(E) @ meu0) - (meu1.T @ inv(E) @ meu1)) / 2 - np.log((1 - phi) / phi)
    z = inv(E) @ (meu1 - meu0)
    z = np.array(z, dtype='f')
    x1 = np.array([25, 200])
    x2 = (-1) * (x1 * z[0] + c) / z[1]
    plt.plot(x1, x2, 'g-')
    if flag == 0:
        plt.show()
    

def GDAd():
    E0 = ((X * (1-Y)[:, np.newaxis] - meu00).T @ (X * (1-Y)[:, np.newaxis] - meu00))  / (M - s)
    E1 = ((X * Y[:, np.newaxis] - meu11).T @ (X * Y[:, np.newaxis] - meu11))  / s
    print("Covarience when E0 <> E1")
    print("E0 : ", E0)
    print("E1 : ", E1)
    E0inv = inv(E0)
    E1inv = inv(E1)
    c = ((meu0.T @ E0inv @ meu0) - (meu1.T @ E1inv @ meu1)) / 2 + np.log(np.linalg.det(E0)/np.linalg.det(E1)) / 2 + np.log(phi / (1 - phi))
    
    Edifference = (E0inv - E1inv)/2
    
    c1 = (meu0.T @ E0inv - meu1.T @ E1inv) * (-1)
    
    X1 = np.linspace(-40, 360, 100)
    X2 = np.linspace(125, 1000, 100)
    x1, x2 = meshgrid(X1, X2)
    Expr = (x1 * x1 * Edifference[0, 0] + (Edifference[0, 1] + Edifference[1, 0]) * x1 * x2 + x2 *x2 * Edifference[1, 1]) + c1[0] * x1 + c1[1] * x2 + c
    plt.contour(x1, x2, Expr, [0], colors='r')
    plt.xlabel('Alaska')
    plt.ylabel('Canada')
    plt.title("Gaussian Discriminant Analysis")
    plt.savefig("Q4_e.png")
    plt.show()
    plt.close()

if(flag == 0):
    GDAa()
if(flag == 1):
    GDAa()
    GDAd()
