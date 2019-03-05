import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

if(len(sys.argv) == 4):
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    tau = float(sys.argv[3])
    
else:
    file1 = "weightedX.csv"
    file2 = "weightedY.csv"
    tau = 0.8
    

Xvalues = np.loadtxt(file1, delimiter = ",")
Mean = np.mean(Xvalues)
StdDev = np.std(Xvalues)
Xvalues = (Xvalues - Mean) / StdDev
X = np.array([[1, x] for x in Xvalues])
Y = np.loadtxt(file2, delimiter = ",")
(M, ) = np.shape(Xvalues)
N = 1
 


# Function for part c
def GraphWithDifferentTau():
    PlotGraph(0.1)
    PlotGraph(0.3)
    PlotGraph(2)
    PlotGraph(10)


def Unweighted():
    Theta = inv(X.T @ X) @ X.T @ Y
    x = np.linspace(-2.5,2.5,100)
    y = Theta[0] + Theta[1] * x
    plt.plot(Xvalues, Y, 'r+')
    plt.plot(x, y, 'g-')
    plt.xlabel(r'$\theta_0$', labelpad=10)
    plt.ylabel(r'$\theta_1$', labelpad=10)
    plt.title("Unweighted Linear Regression")
    plt.savefig("Q2_a.png")
    plt.show()
    plt.close()
    # print(Theta)


def WeightedMatrix(Tau, x):
    Weight = []
    for r in X:
        Weight.append(np.exp((-1) * ((x - r[1]) ** 2 / (2 * Tau * Tau))))
    return Weight
    
def Weighted(Tau, x):
    Weight = WeightedMatrix(Tau, x)
    # print(Weight, np.shape(Weight))
    Weight = np.diag(Weight)
    Theta = inv(X.T @ Weight @ X) @ (X.T @ Weight @ Y)
    # print(Theta)
    x = np.linspace(x-0.05,x+0.05,3)
    y = Theta[0] + Theta[1] * x
    plt.plot(x, y, 'g-')

#Function for part b    
def PlotGraph(Tau):
    plt.figure()
    plt.plot(Xvalues, Y, 'r+')
    xdatapts = np.linspace(np.min(Xvalues), np.max(Xvalues), 100)
    for xc in xdatapts:
        Weighted(Tau, xc)
    plt.xlabel(r'$\theta_0$', labelpad=10)
    plt.ylabel(r'$\theta_1$', labelpad=10)
    plt.title("Weighted Linear Regression with Tau : " + str(Tau))
    plt.savefig("Q2_b" + str(Tau) + ".png")
    plt.show()
    plt.close()

Unweighted()

PlotGraph(tau)

GraphWithDifferentTau()