
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

#(a) Plot a picture of the mean vector.
mean = sum(digits.images)/digits.images.shape[0]
plt.imshow(mean, cmap=plt.cm.gray_r)
plt.show()

mu = mean.flatten()
N = digits.images.shape[0]
D = digits.images.shape[1]*digits.images.shape[2]
H = 2

X = np.zeros((D, N))
for i in range(N-1):
    X[:, i] = digits.images[i].flatten() - mu

#initalize the loading matrix and plot(max=1, min=0)
W = np.random.uniform(0,1,(D,H))
plt.imshow(W, cmap=plt.cm.gray_r)
plt.show()

#initalize diagonal elements of noise variance
Psi = np.diag(np.random.uniform(0,1,D))
plt.imshow(Psi, cmap=plt.cm.gray_r)
plt.show()


Z = np.zeros((N,H))

#Calculate the expectation of z|x
def Expect_z_x(W, Psi, x):
    Psi_inv= np.linalg.pinv(Psi)
    return np.dot(np.linalg.pinv((np.identity(H) + np.dot(np.dot(W.T,Psi_inv), W))), np.dot(np.dot(W.T, Psi_inv), x))


#calculate z from X = Wz + eplison
def Update_z(W, Psi, X, Z):
    for i in range(N - 1):
        Z[i, :] = np.dot(np.linalg.pinv(W), X[:, i] - np.random.multivariate_normal(np.zeros(D), Psi).T)
    return Z

#calculate the log-likelihood function
def Loglikelihood(W, Psi, X, Z):
    l=0
    Z = Update_z(W, Psi, X, Z)
    for i in range(0, N-1):
        l += np.dot(np.dot((X[:, i]-np.dot(W, Z[i, :])).T,np.linalg.pinv(Psi)), (X[:, i]-np.dot(W, Z[i, :]))) + np.dot(Z[i, :].T, Z[i, :])
    l *= -0.5
    l -= N/2*(np.log(np.linalg.det(Psi))-np.log(H))-N*D/2*np.log(2*np.pi)
    return l

#Iteration for W
def Iteration_W(W, Psi, X):
    temp1 = np.zeros((D, H))
    temp2 = np.zeros((H, H))
    for i in range(0, N-1):
        Ez_x = Expect_z_x(W, Psi, X[:, i]).reshape(H,-1)
        temp1 += np.dot(X[:, i].reshape(D,-1), Ez_x.T)
        temp2 += np.dot(Ez_x, Ez_x.T)
    return np.dot(temp1, np.linalg.inv(temp2))

#Iteration for Phi
def Iteration_Psi(W, Psi, X):
    temp = np.zeros((64,64))
    for i in range(0, N-1):
        Ez_x = Expect_z_x(W, Psi, X[:, i])
        print(Ez_x)
        temp += np.dot(X[:, i].reshape(D,-1), X[:,i].reshape(D,-1).T) + np.dot(W, np.dot(Ez_x.reshape(H,1), X[:,i].reshape(D,-1).T))
    return np.diag(np.diag(temp))/N

l_old = 0
l_new = 500

while l_new - l_old > 1:
    l_old = l_new
    W = Iteration_W(W, Psi, X)
    Psi = Iteration_Psi(W, Psi, X)
    l_new = Loglikelihood(W, Psi, X, Z)
    print(l_new)
