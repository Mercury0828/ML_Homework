
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from scipy.linalg import sqrtm


digits = datasets.load_digits()

#(a) Plot a picture of the mean vector.
mean = sum(digits.images)/digits.images.shape[0]
plt.imshow(mean, cmap=plt.cm.gray_r)
plt.show()

mu = mean.flatten()
N = digits.images.shape[0]
D = digits.images.shape[1]*digits.images.shape[2]
H = 2

#initalize the loading matrix and plot(max=1, min=0)
W = np.random.uniform(0,1,(D,H))
plt.imshow(W, cmap=plt.cm.gray_r)
plt.show()

#initalize diagonal elements of noise variance
Psi = np.diag(np.random.uniform(0,1,D))
plt.imshow(Psi, cmap=plt.cm.gray_r)
plt.show()


X = np.zeros((D, N))
for i in range(N-1):
    X[:, i] = digits.images[i].flatten() - mu

Var = np.zeros(D)
for i in range(D-1):
    Var[i] = np.var(X[i, :])

def Scale_X(X, Psi):
    return np.dot(np.linalg.pinv(sqrtm(Psi)), X)/ np.sqrt(N)

do_svd = lambda x: np.linalg.svd(x)



l_old = 0
l_new = 500


#using the algorithm 12.41
while abs(l_new - l_old) > 1:
    l_old = l_new
    X = Scale_X(X, Psi)
    U, A, W = do_svd(X)
    A = np.diag(A)
    A = np.dot(A, A)
    U_h = U[:, 0:H]
    A_h = np.diag(np.diag(A)[0:H])
    F = np.dot(np.dot(sqrtm(Psi), U_h), sqrtm(A_h - np.identity(H)))
    l_new = -N/2*sum(np.log(np.diag(A)[0:H]))+H+sum(np.diag(A)[H+1:D]) + np.log(np.linalg.det(2*np.pi*Psi))
    print(l_new)
    Psi = np.diag(Var)-np.diag(np.dot(F,F.T))
