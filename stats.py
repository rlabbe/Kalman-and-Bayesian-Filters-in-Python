"""
Author: Roger Labbe
Copyright: 2014

This code performs various basic statistics functions for the
Kalman and Bayesian Filters in Python book. Much of this code
is non-optimal; production code should call the equivalent scipy.stats
functions. I wrote the code in this form to make explicit how the
computations are done. The scipy.stats module has many more useful functions
than what I have written here. In some cases, however, my code is significantly
faster, at least on my machine. For example, gaussian average 794 ns, whereas
stats.norm(), using the frozen form, averages 116 microseconds per call.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.stats
from matplotlib.patches import Ellipse

_two_pi = 2*math.pi


def gaussian(x, mean, var):
    """returns normal distribution for x given a gaussian with the specified
    mean and variance. All must be scalars.

    gaussian (1,2,3) is equivalent to scipy.stats.norm(2,math.sqrt(3)).pdf(1)
    It is quite a bit faster albeit much less flexible than the latter.
    """
    return math.exp((-0.5*(x-mean)**2)/var) / math.sqrt(_two_pi*var)
    # return scipy.stats.norm(mean, math.sqrt(var)).pdf(x)


def mul (mean1, var1, mean2, var2):
    """ multiply Gaussians (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean,var).

    var1 and var2 are variances - sigma squared in the usual parlance.
    """

    mean = (var1*mean2 + var2*mean1) / (var1 + var2)
    var = 1 / (1/var1 + 1/var2)
    return (mean, var)


def add (mean1, var1, mean2, var2):
    """ add the Gaussians (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean,var).

    var1 and var2 are variances - sigma squared in the usual parlance.
    """

    return (mean1+mean2, var1+var2)


def multivariate_gaussian(x, mu, cov):
    """ This is designed to replace scipy.stats.multivariate_normal
    which is not available before version 0.14. You may either pass in a
    multivariate set of data:
       multivariate_gaussian (array([1,1]), array([3,4]), eye(2)*1.4)
       multivariate_gaussian (array([1,1,1]), array([3,4,5]), 1.4)
    or unidimensional data:
       multivariate_gaussian(1, 3, 1.4)

    In the multivariate case if cov is a scalar it is interpreted as eye(n)*cov

    The function gaussian() implements the 1D (univariate)case, and is much
    faster than this function.

    equivalent calls:
       multivariate_gaussian(1, 2, 3)
       scipy.stats.multivariate_normal(2,3).pdf(1)
    """

    # force all to numpy.array type
    x   = np.array(x, copy=False, ndmin=1)
    mu  = np.array(mu,copy=False, ndmin=1)

    nx = len(mu)
    cov = _to_cov(cov, nx)

    norm_coeff = nx*math.log(2*math.pi) + np.linalg.slogdet(cov)[1]

    err = x - mu
    if (sp.issparse(cov)):
        numerator = spln.spsolve(cov, err).T.dot(err)
    else:
        numerator = np.linalg.solve(cov, err).T.dot(err)

    return math.exp(-0.5*(norm_coeff + numerator))


def plot_gaussian(mean, variance,
                  mean_line=False,
                  xlim=None,
                  xlabel=None,
                  ylabel=None):
    """ plots the normal distribution with the given mean and variance. x-axis
    contains the mean, the y-axis shows the probability.

    mean_line : draws a line at x=mean
    xlim: optionally specify the limits for the x axis as tuple (low,high).
          If not specified, limits will be automatically chosen to be 'nice'
    xlabel : optional label for the x-axis
    ylabel : optional label for the y-axis
    """

    sigma = math.sqrt(variance)
    n = scipy.stats.norm(mean, sigma)

    if xlim is None:
        min_x = n.ppf(0.001)
        max_x = n.ppf(0.999)
    else:
        min_x = xlim[0]
        max_x = xlim[1]
    xs = np.arange(min_x, max_x, (max_x - min_x) / 1000)
    plt.plot(xs,n.pdf(xs))
    plt.xlim((min_x, max_x))

    if mean_line:
        plt.axvline(mean)
    if xlabel:
       plt.xlabel(xlabel)
    if ylabel:
       plt.ylabel(ylabel)


def covariance_ellipse(P):
    """ returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Returns (angle_radians, width_radius, height_radius)
    """
    U,s,v = linalg.svd(P)
    orientation = math.atan2(U[1,0],U[0,0])
    width  = math.sqrt(s[0])
    height = math.sqrt(s[1])
    return (orientation, width, height)


def plot_covariance_ellipse(mean, cov=None, variance = 1.0,
             ellipse=None, title=None, axis_equal=True,
             facecolor='none', edgecolor='blue'):
    """ plots the covariance ellipse where

    mean is a (x,y) tuple for the mean of the covariance (center of ellipse)

    cov is a 2x2 covariance matrix.

    variance is the normal sigma^2 that we want to plot. If list-like,
    ellipses for all ellipses will be ploted. E.g. [1,2] will plot the
    sigma^2 = 1 and sigma^2 = 2 ellipses.

    ellipse is a (angle,width,height) tuple containing the angle in radians,
    and width and height radii.

    You may provide either cov or ellipse, but not both.

    plt.show() is not called, allowing you to plot multiple things on the
    same figure.
    """

    assert cov is None or ellipse is None
    assert not (cov is None and ellipse is None)

    if cov is not None:
        ellipse = covariance_ellipse(cov)

    if axis_equal:
        plt.axis('equal')

    if title is not None:
        plt.title (title)


    if np.isscalar(variance):
        variance = [variance]

    ax = plt.gca()

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.
    height = ellipse[2] * 2.

    for var in variance:
        e = Ellipse(xy=mean, width=var*width, height=var*height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    lw=1)
        ax.add_patch(e)
    plt.scatter(mean[0], mean[1], marker='+') # mark the center


def _to_cov(x,n):
    """ If x is a scalar, returns a covariance matrix generated from it
    as the identity matrix multiplied by x. The dimension will be nxn.
    If x is already a numpy array then it is returned unchanged.
    """
    try:
        x.shape
        if type(x) != np.ndarray:
            x = np.asarray(x)[0]
        return x
    except:
        return np.eye(n) * x


def test_gaussian():
   import scipy.stats

   mean = 3.
   var = 1.5
   std = var*0.5

   for i in np.arange(-5,5,0.1):
       p0 = scipy.stats.norm(mean, std).pdf(i)
       p1 = gaussian(i, mean, var)

       assert abs(p0-p1) < 1.e15


if __name__ == '__main__':

    from scipy.stats import norm

    test_gaussian()

    # test conversion of scalar to covariance matrix
    x  = multivariate_gaussian(np.array([1,1]), np.array([3,4]), np.eye(2)*1.4)
    x2 = multivariate_gaussian(np.array([1,1]), np.array([3,4]), 1.4)
    assert x == x2

    # test univarate case
    rv = norm(loc = 1., scale = np.sqrt(2.3))
    x2 = multivariate_gaussian(1.2, 1., 2.3)
    x3 = gaussian(1.2, 1., 2.3)

    assert rv.pdf(1.2) == x2
    assert abs(x2- x3) < 0.00000001

    cov = np.array([[1.0, 1.0],
                    [1.0, 1.1]])

    P = np.array([[2,0],[0,2]])
    plot_covariance_ellipse((2,7), cov=cov, variance=[1,2], title='my title')
    plt.show()

    print("all tests passed")
