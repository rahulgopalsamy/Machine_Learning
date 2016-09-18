import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi, exp
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import math


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    # Meaan
    # k classes
    k = np.unique(y)
    means = np.zeros([k.shape[0], X.shape[1]])
    Y = y.flatten();
    count = np.zeros([k.shape[0]])
    # Sum of elements of each class
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            index = Y[i] - 1
            means[index][j] = means[index][j] + X[i][j]
        count[index] = count[index] + 1
    # Dividing by count of each class to get mean
    for i in range(k.shape[0]):
        for j in range(X.shape[1]):
            means[i][j] = means[i][j] / count[i]
    means = means.T
    # Covariance
    covmat = np.cov(X, rowvar=0)
    # print ("LDA mean", means)
    # print ("LDA cov", covmat)
    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    k = np.unique(y)
    means = np.zeros([k.shape[0], X.shape[1]])
    Y = y.flatten();
    count = np.zeros([k.shape[0]])
    # Sum of elements of each class
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            index = Y[i] - 1
            means[index][j] = means[index][j] + X[i][j]
        count[index] = count[index] + 1
    # Dividing by count of each class to get mean
    for i in range(k.shape[0]):
        for j in range(X.shape[1]):
            means[i][j] = means[i][j] / count[i]
    means = means.T
    # Covariance
    covmats = [np.zeros((X.shape[1], X.shape[1]))] * k.shape[0]
    for i in range(k.size):
        X_class = X[Y == k[i]]
        covmats[i] = np.cov(X_class, rowvar=0)
    # print covmats
    # print ("QDA mean", means)
    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    means = means.T
    k = np.unique(y)
    Ytest = ytest.flatten()
    ypred = np.zeros([Xtest.shape[0]])
    covmat_det = sqrt(np.linalg.det(covmat))
    covmat_inv = inv(covmat)
    acc = 0
    for a in range(Xtest.shape[0]):
        result = np.zeros([k.shape[0]])
        for i in range(k.shape[0]):
            X_mean = Xtest[a] - means[i]
            epow = -1 / 2 * np.dot(np.dot(X_mean, covmat_inv), X_mean.T)
            result[i] = (1 / (sqrt(2 * pi) * covmat_det)) * (math.exp(epow))
        ypred[a] = float(result.argmax(axis=0) + 1)
        if (ypred[a] == Ytest[a]):
            acc = acc + 1;
    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    means = means.T
    k = np.unique(y)
    Ytest = ytest.flatten()
    ypred = np.zeros([Xtest.shape[0]])
    acc = 0
    for a in range(Xtest.shape[0]):
        result = np.zeros([k.shape[0]])
        for i in range(k.shape[0]):
            covmat_det = sqrt(np.linalg.det(covmats[i]))
            covmat_inv = inv(covmats[i])
            X_mean = Xtest[a] - means[i]
            epow = -1 / 2 * np.dot(np.dot(X_mean, covmat_inv), X_mean.T)
            result[i] = (1 / (sqrt(2 * pi) * covmat_det)) * (math.exp(epow))
        ypred[a] = float(result.argmax(axis=0) + 1)
        if (ypred[a] == Ytest[a]):
            acc = acc + 1;
    return acc, ypred


def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # From slide
    # IMPLEMENT THIS METHOD
    w = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))
    return w


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1
    # From slide
    # IMPLEMENT THIS METHOD
    XT_X = np.dot(X.T, X)
    lambda_mat = lambd * np.identity(XT_X.shape[0])
    w = np.dot(inv((lambda_mat) + XT_X), np.dot(X.T, y))
    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    # IMPLEMENT THIS METHOD
    squ = np.square(ytest - np.dot(Xtest, w))
    rmse = sqrt(np.sum(squ) / Xtest.shape[0])
    return rmse


def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    w = np.mat(w)
    w = w.T
    wt_x = np.dot(X, w)
    error = (np.dot((y - wt_x).T, (y - wt_x))) / (2) + (lambd * np.dot(w.T, w) / 2)
    term1 = np.dot(w.T, X.T) - y.T
    # print ("term1", term1.shape)
    lambda1 = lambd * w.T
    gradient_des = np.dot(term1, X) + lambda1
    error_grad = np.array(gradient_des).flatten()
    return error, error_grad


def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD
    Xd = np.zeros([x.shape[0], p + 1])
    for i in range(x.shape[0]):
        for j in range(p + 1):
            Xd[i][j] = np.power(x[i], j)
    return Xd


# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
ldaacc = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = ' + str(ldaacc))
# QDA
means, covmats = qdaLearn(X, y)
qdaacc = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = ' + str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])))
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.show()

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])))
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.show()
# Problem 2

if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)
# mle = testOLERegression(w, X, y)
w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)
# mle_i = testOLERegression(w_i, X_i, y)

print('RMSE without intercept ' + str(mle))
print('RMSE with intercept ' + str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    rmses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    # rmses3[i] = testOLERegression(w_l, X_i, y)
    i = i + 1
plt.plot(lambdas, rmses3)
plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k, 1))
opts = {'maxiter': 100}  # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    rmses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    # rmses4[i] = testOLERegression(w_l, X_i, y)
    i = i + 1
plt.plot(lambdas, rmses4)
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    rmses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    rmses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)
plt.plot(range(pmax), rmses5)
plt.legend(('No Regularization', 'Regularization'))
plt.show()
