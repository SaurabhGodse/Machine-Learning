import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation

if(len(sys.argv) == 5):
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    eta = float(sys.argv[3])
    time = float(sys.argv[4])
else:
    file1 = "linearX.csv"
    file2 = "linearY.csv"
    eta = 0.1
    time = 0.2



Xvalues = np.loadtxt(file1, delimiter = ",")

Mean = np.mean(Xvalues)
StdDev = np.std(Xvalues)


Y = np.loadtxt(file2, delimiter = ",")
(M, ) = np.shape(Xvalues)
N = 1

Xvalues = (Xvalues - Mean) / StdDev
   
X = [[1,x] for x in Xvalues]

def ContoursForVariousEta():
	Eta = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.3]
	for e in Eta:
		# print("For e = ", e)
		MeshPoints, e = GDOptimization(e)
		Contours(MeshPoints, e)



def ThreeDMesh(MeshPoints, time):
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    X = np.arange(-1, 3, 0.04)
    Y = np.arange(-2, 2, 0.04)
    X, Y = np.meshgrid(X, Y)
    Xf = X.flatten()
    Yf = Y.flatten()
    Z = []
    
    Mesh = np.c_[Xf, Yf]
    
    for r in Mesh:
        Z.append(costofJ(r))
    Z = np.reshape(Z, np.shape(X))
   
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    
    ax.set_xlabel(r'$\theta_0$', labelpad=5)
    ax.set_ylabel(r'$\theta_1$', labelpad=5)
    ax.set_zlabel(r'$J(\theta)$', labelpad=5)
    fig.colorbar(surf, shrink=0.5, aspect=6)
    
    for r in MeshPoints:
        ax.plot([r[0]], [r[1]], [r[2]], linestyle='-',color='m', marker='x', markersize=4)
        plt.pause(time)
    plt.show()
    plt.title("Three D Mesh")
    plt.savefig("Q1_c.png")
    plt.close()



def Contours(MeshPoints, eta):
    X = np.arange(-1, 3, 0.04)
    Y = np.arange(-2, 2, 0.04)
    
    X, Y = np.meshgrid(X, Y)
    Xf = X.flatten()
    Yf = Y.flatten()
    Z = []
    Mesh = np.c_[Xf, Yf]
    for r in Mesh:
        Z.append(costofJ(r))
    Z = np.reshape(Z, np.shape(X))
    plt.ion()
    plt.contour(X, Y, Z, 25)
    plt.xlabel(r'$\theta_0$', labelpad=10)
    plt.ylabel(r'$\theta_1$', labelpad=10)
    plt.xlim(0, 2)
    plt.ylim(-2, 2)
    plt.show()
    for r in MeshPoints:
        plt.plot([r[0]], [r[1]], linestyle='-',color='r', marker='x', markersize=4)
        plt.pause(0.2)
    plt.title(r"$Contours (\eta=$" + str(eta) + ")")
    plt.show()
    plt.savefig(f"Q1_d :{eta}.png")
    plt.close()



def costofJ(Theta):
	return np.sum((X @ Theta - Y) ** 2 / (2 * M))

def GDOptimization(eta):
    Theta = np.zeros(N + 1)
    print("Eta = ", eta)
    # print("Theta :", Theta)
    PrevCost = costofJ(Theta)
    iterations = 0
    MeshPoints = np.array([[Theta[0], Theta[1], PrevCost]])
    IsConverged = False
    epsilon = 10 ** (-9)
    print("Stopping Criteria : ", epsilon)
    while(not IsConverged):
        Theta = Theta - eta * ((X @ Theta - Y) @ X) / M
        CurrentCost = costofJ(Theta);
        MeshPoints = np.append(MeshPoints, [[Theta[0], Theta[1], CurrentCost]], axis=0)
        if(abs(CurrentCost - PrevCost) <= epsilon):
            print("Converged Successfully")
            IsConverged = True
        
        iterations += 1
        PrevCost = CurrentCost
        if(iterations == 10000):
            print("Too much iterations")
            break
    x = np.linspace(-5,5,100)
    y = Theta[0] + Theta[1] * x
    plt.plot(Xvalues, Y, 'r+')
    plt.plot(x, y, 'g-')
    plt.xlabel(r'$\theta_0$', labelpad=10)
    plt.ylabel(r'$\theta_1$', labelpad=10)

    print("Minimum cost : ", CurrentCost)
    print("Final value of Theta :", Theta)
    print("Iterations :", iterations)
    plt.title("Plot of hypothesis function")
    plt.savefig("Q1_b.png")
    plt.show()
    
    plt.close()
    return MeshPoints, eta
  
     
MeshPoints, eta = GDOptimization(eta)
ThreeDMesh(MeshPoints, time)
Contours(MeshPoints, eta)
ContoursForVariousEta()

