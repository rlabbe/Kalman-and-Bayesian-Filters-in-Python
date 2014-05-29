# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:40:19 2014

@author: rlabbe
"""

from numpy import matrix, zeros, asmatrix, size
from  numpy.linalg import cholesky


def sigma_points (mu, cov, kappa):
    """ Computes the sigma points and weights for an unscented Kalman filter.
    xm are the means, and P is the covariance. kappa is an arbitrary constant
    constant. Returns tuple of the sigma points and weights.

    This is the original algorithm as published by Julier and Uhlmann.
    Later algorithms introduce more parameters - alpha, beta,

    Works with both scalar and array inputs:
    sigma_points (5, 9, 2) # mean 5, covariance 9
    sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
    """

    mu  = asmatrix(mu)
    cov = asmatrix(cov)

    n = size(mu)

    # initialize to zero
    Xi = asmatrix (zeros((n,2*n+1)))
    W  = asmatrix (zeros(2*n+1))


    # all weights are 1/ 2(n+kappa)) except the first one.
    W[0,1:] = 1. / (2*(n+kappa))
    W[0,0] = float(kappa) / (n + kappa)


    # use cholesky to find matrix square root of (n+kappa)*cov
    #  U'*U = (n+kappa)*P
    U = asmatrix (cholesky((n+kappa)*cov))

    # mean is in location 0.
    Xi[:,0] = mu

    for k in range (n):
        Xi[:,k+1] = mu + U[:,k]

    for k in range (n):
        Xi[:, n+k+1] = mu - U[:,k]

    return (Xi, W)



def unscented_transform (Xi, W, NoiseCov=None):
    """ computes the unscented transform of a set of signma points and weights.
    returns the mean and covariance in a tuple
    """
    W = asmatrix(W)
    Xi = asmatrix(Xi)

    n, kmax = Xi.shape

    # initialize results to 0
    mu  = matrix (zeros((n,1)))
    cov = matrix (zeros((n,n)))

    for k in range (kmax):
        mu += W[0,k] * Xi[:,k]

    for k in range (kmax):
        cov += W[0,k]*(Xi[:,k]-mu) * (Xi[:,k]-mu).T

    return (mu, cov)


xi = matrix ([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
mu = matrix ([1,2,3,4,5,6,7])

m,c = unscented_transform(xi, mu)
print m
print c
