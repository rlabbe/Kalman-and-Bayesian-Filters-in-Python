# -*- coding: utf-8 -*-
"""

py.test module to test stats.py module.


Created on Wed Aug 27 06:45:06 2014

@author: rlabbe
"""
from __future__ import division
from math import pi, exp
import numpy as np
from stats import gaussian, multivariate_gaussian, _to_cov
from numpy.linalg import inv
from numpy import linalg


def near_equal(x,y):
    return abs(x-y) < 1.e-15


def test_gaussian():
    import scipy.stats

    mean = 3.
    var = 1.5
    std = var**0.5

    for i in np.arange(-5,5,0.1):
        p0 = scipy.stats.norm(mean, std).pdf(i)
        p1 = gaussian(i, mean, var)

        assert near_equal(p0, p1)



def norm_pdf_multivariate(x, mu, sigma):
    """ extremely literal transcription of the multivariate equation.
    Slow, but easy to verify by eye compared to my version."""

    n = len(x)
    sigma = _to_cov(sigma,n)

    det = linalg.det(sigma)

    norm_const = 1.0 / (pow((2*pi), n/2) * pow(det, .5))
    x_mu = x - mu
    result = exp(-0.5 * (x_mu.dot(inv(sigma)).dot(x_mu.T)))
    return norm_const * result



def test_multivariate():
    from scipy.stats import multivariate_normal as mvn
    from numpy.random import rand

    mean = 3
    var = 1.5

    assert near_equal(mvn(mean,var).pdf(0.5),
                      multivariate_gaussian(0.5, mean, var))

    mean = np.array([2.,17.])
    var = np.array([[10., 1.2], [1.2, 4.]])

    x = np.array([1,16])
    assert near_equal(mvn(mean,var).pdf(x),
                      multivariate_gaussian(x, mean, var))

    for i in range(100):
        x = np.array([rand(), rand()])
        assert near_equal(mvn(mean,var).pdf(x),
                          multivariate_gaussian(x, mean, var))

        assert near_equal(mvn(mean,var).pdf(x),
                          norm_pdf_multivariate(x, mean, var))


    mean = np.array([1,2,3,4])
    var = np.eye(4)*rand()

    x = np.array([2,3,4,5])

    assert near_equal(mvn(mean,var).pdf(x),
                      norm_pdf_multivariate(x, mean, var))


