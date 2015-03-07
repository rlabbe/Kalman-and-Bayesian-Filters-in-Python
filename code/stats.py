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
from scipy.stats import norm
from matplotlib.patches import Ellipse

_two_pi = 2*math.pi


def gaussian(x, mean, var):
    """returns normal distribution (pdf) for x given a Gaussian with the
    specified mean and variance. x can either be a scalar or an 
    array-like. 

    gaussian (1,2,3) is equivalent to scipy.stats.norm(2,math.sqrt(3)).pdf(1)
    It is quite a bit faster albeit much less flexible than the latter.

    @param x test

    Parameters
    ----------
    x : scalar or array-like
        The value for which we compute the probability

    mean : scalar
        Mean of the Gaussian

    var : scalar
        Variance of the Gaussian

    Returns
    -------
    probability : float, or array-like
        probability of x for the Gaussian (mean, var). E.g. 0.101 denotes
        10.1%.
        
    Examples
    --------
    gaussian(3, 1, 2)
    gaussian([3,4,3,2,1], 1, 2)
    """
    
    return (np.exp((-0.5*(np.asarray(x)-mean)**2)/var) / 
            np.sqrt(_two_pi*var))


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



    Parameters
    ----------
    x : scalar, or np.array-like
       Value to compute the probability for. May be a scalar if univariate,
       or any type that can be converted to an np.array (list, tuple, etc).
       np.array is best for speed.

    mu :  scalar, or np.array-like
       mean for the Gaussian . May be a scalar if univariate,  or any type
       that can be converted to an np.array (list, tuple, etc).np.array is
       best for speed.

    cov :  scalar, or np.array-like
       Covariance for the Gaussian . May be a scalar if univariate,  or any
       type that can be converted to an np.array (list, tuple, etc).np.array is
       best for speed.



    Returns
    -------
    probability : float
        probability for x for the Gaussian (mu,cov)


    """
    scipy.stats.multivariate_normal
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


def covariance_ellipse(P, deviations=1):
    """ returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters
    ----------
    P : nd.array shape (2,2)
       covariance matrix

    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.

    Returns (angle_radians, width_radius, height_radius)
    """
    U,s,v = linalg.svd(P)
    orientation = math.atan2(U[1,0],U[0,0])
    width  = deviations*math.sqrt(s[0])
    height = deviations*math.sqrt(s[1])

    assert width >= height

    return (orientation, width, height)


def is_inside_ellipse(x,y, ex, ey, orientation, width, height):

    co = np.cos(orientation)
    so = np.sin(orientation)

    xx = x*co + y*so
    yy = y*co - x*so

    return (xx / width)**2 + (yy / height)**2 <= 1.


    return ((x-ex)*co - (y-ey)*so)**2/width**2 + \
           ((x-ex)*so + (y-ey)*co)**2/height**2 <= 1



def plot_covariance_ellipse(mean, cov=None, variance = 1.0,
             ellipse=None, title=None, axis_equal=True,
             facecolor='none', edgecolor='#004080',
             alpha=1.0, xlim=None, ylim=None):
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
        sd = np.sqrt(var)
        e = Ellipse(xy=mean, width=sd*width, height=sd*height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    lw=2)
        ax.add_patch(e)
    plt.scatter(mean[0], mean[1], marker='+') # mark the center
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)


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



def do_plot_test():

    from numpy.random import multivariate_normal
    p = np.array([[32, 15],[15., 40.]])

    x,y = multivariate_normal(mean=(0,0), cov=p, size=5000).T
    sd = 2
    a,w,h = covariance_ellipse(p,sd)
    print (np.degrees(a), w, h)

    count = 0
    color=[]
    for i in range(len(x)):
        if is_inside_ellipse(x[i], y[i], 0, 0, a, w, h):
            color.append('b')
            count += 1
        else:
            color.append('r')
    plt.scatter(x,y,alpha=0.2, c=color)


    plt.axis('equal')

    plot_covariance_ellipse(mean=(0., 0.),
                            cov = p,
                            variance=sd*sd,
                            facecolor='none')

    print (count / len(x))


from numpy.linalg import inv
from numpy import asarray, dot


def multivariate_multiply(m1, c1, m2, c2):

    C1 = asarray(c1)
    C2 = asarray(c2)
    M1 = asarray(m1)
    M2 = asarray(m2)

    sum_inv = inv(C1+C2)
    C3 = dot(C1, sum_inv).dot(C2)

    M3 = (dot(C2, sum_inv).dot(M1) +
          dot(C1, sum_inv).dot(M2))

    return M3, C3

def norm_cdf (x_range, mu, var=1, std=None):
    """ computes the probability that a Gaussian distribution lies
    within a range of values.

    Paramateters
    ------------

    x_range : (float, float)
        tuple of range to compute probability for

    mu : float
        mean of the Gaussian

    var : float, optional
        variance of the Gaussian. Ignored if std is provided

    std : float, optional
       standard deviation of the Gaussian. This overrides the var parameter


    Returns
    -------
    probability : float
        probability that Gaussian is within x_range. E.g. .1 means 10%.

    """

    if std is None:
        std = math.sqrt(var)
    return abs(norm.cdf(x_range[0], loc=mu, scale=std) -
               norm.cdf(x_range[1], loc=mu, scale=std))



def test_norm_cdf():

    # test using the 68-95-99.7 rule

    mu = 5
    std = 3
    var = std*std

    std_1 = (norm_cdf((mu-std, mu+std), mu, var))
    assert abs(std_1 - .6827) < .0001

    std_1 = (norm_cdf((mu+std, mu-std), mu, std=std))
    assert abs(std_1 - .6827) < .0001

    std_1half = (norm_cdf((mu+std, mu), mu, var))
    assert abs(std_1half - .6827/2) < .0001

    std_2 = (norm_cdf((mu-2*std, mu+2*std), mu, var))
    assert abs(std_2 - .9545) < .0001

    std_3 = (norm_cdf((mu-3*std, mu+3*std), mu, var))
    assert abs(std_3 - .9973) < .0001


if __name__ == '__main__':
    test_norm_cdf ()

    do_plot_test()

    #test_gaussian()

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

    plt.figure()
    P = np.array([[2,0],[0,2]])
    plot_covariance_ellipse((2,7), cov=cov, variance=[1,2], facecolor='g', title='my title', alpha=.2)
    plt.show()

    print("all tests passed")
