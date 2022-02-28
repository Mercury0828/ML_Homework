
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from scipy.linalg import sqrtm

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

#initalize the loading matrix and plot(max=1, min=0)
W = np.random.uniform(0,1,(D,H))
plt.imshow(W, cmap=plt.cm.gray_r)
plt.show()


Images = np.zeros((D, N))
for i in range(N):
    Images[:, i] = digits.images[i].flatten() - mu

Var = np.zeros(D)
for i in range(D):
    Var[i] = np.var(Images[i, :])
    if Var[i] == 0:
        Var[i] = 0.01


#initalize diagonal elements of noise variance
Psi = np.diag(Var)
plt.imshow(Psi, cmap=plt.cm.gray_r)
plt.show()


def Scale_X(X, Psi):
    return np.dot(np.linalg.pinv(sqrtm(Psi)), X)/ np.sqrt(N)

do_svd = lambda x: np.linalg.svd(x)

def Llikelihood(Psi, X):
    X = Scale_X(X, Psi)
    U, A, W = do_svd(X)
    A = np.diag(A)
    A = np.dot(A, A)
    U_h = U[:, 0:H]
    A_h = np.diag(np.diag(A)[0:H])
    F = np.dot(np.dot(sqrtm(Psi), U_h), sqrtm(A_h - np.identity(H)))
    ll = -N / 2 * sum(np.log(np.diag(A)[0:H])) + H + sum(np.diag(A)[H + 1:D]) + np.log(np.linalg.det(2 * np.pi * Psi))
    return ll, F

def DoIteration(Psi, X):
    l = []
    l.append(0)
    l_old = l[0]
    l_new = 500

    # using the algorithm 12.41
    while abs((l_new - l_old) / l_new) > 0.01:
        l_old = l_new
        l_new, F = Llikelihood(Psi, train_set)
        l.append(abs(l_new))
        print(abs((l_new - l_old) / l_new))
        Psi = np.diag(Var) - np.diag(np.diag(np.dot(F, F.T)))
    return l, F, Psi

def SplitData(X, size):
    Index = np.arange(0,N)
    train_in, test_in = train_test_split(Index, test_size=size, random_state=42)
    train = Images[:, train_in]
    test = Images[:, test_in]
    return train, test

#Calculate the expectation of z|x
def Expect_z_x(F, Psi, x):
    Psi_inv= np.linalg.pinv(Psi)
    return np.dot(np.linalg.pinv((np.identity(H) + np.dot(np.dot(F.T,Psi_inv), F))), np.dot(np.dot(F.T, Psi_inv), x))


train_set, test_set = SplitData(Images, 0.8)
l, F, Psi = DoIteration(Psi, train_set)
plt.plot(l)
plt.show()

Z = np.zeros((test_set.shape[1], 2))
for i in range(test_set.shape[1]):
    Z[i, :] = Expect_z_x(F, Psi, test_set[:, i])

plt.scatter(Z[:,0], Z[:, 1])
plt.show()



