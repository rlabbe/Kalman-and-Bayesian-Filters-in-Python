import numpy as np
import math


def _to_array(x):
    """ returns any of a scalar, matrix, or array as a 1D numpy array
    Example:
       _to_array(3) == array([3])
    """
    try:
        x.shape
        if type(x) != np.ndarray:
            x = np.asarray(x)[0]
        return x
    except:
        return np.array(np.mat(x)).reshape(1)

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

_two_pi = 2*math.pi

def gaussian (x, mean, var):
    """returns normal distribution for x given a gaussian with the specified
    mean and variance. All must be scalars
    """
    return math.exp((-0.5*(x-mean)**2)/var) / math.sqrt(_two_pi*var)


def multivariate_gaussian (x, mu, cov):
    """ This is designed to work the same as scipy.stats.multivariate_normal
    which is not available before version 0.14. You may either pass in a
    multivariate set of data:
       multivariate_gaussian (array([1,1]), array([3,4]), eye(2)*1.4)
       multivariate_gaussian (array([1,1,1]), array([3,4,5]), 1.4)
    or unidimensional data:
       multivariate_gaussian(1, 3, 1.4)

    In the multivariate case if cov is a scalar it is interpreted as eye(n)*cov

    The function gaussian() implements the 1D (univariate)case, and is much
    faster than this function.
    """

    # force all to numpy.array type
    x = _to_array(x)
    mu = _to_array(mu)
    n = mu.size
    cov = _to_cov (cov, n)

    det = np.sqrt(np.prod(np.diag(cov)))
    frac = _two_pi**(-n/2.) * (1./det)
    fprime = (x - mu)**2
    return frac * np.exp(-0.5*np.dot(fprime, 1./np.diag(cov)))


if __name__ == '__main__':
    from scipy.stats import norm

    # test conversion of scalar to covariance matrix
    x  = multivariate_gaussian(np.array([1,1]), np.array([3,4]), np.eye(2)*1.4)
    x2 = multivariate_gaussian(np.array([1,1]), np.array([3,4]), 1.4)
    assert x == x2

    # test univarate case
    rv = norm (loc = 1., scale = np.sqrt(2.3))
    x2 = multivariate_gaussian (1.2, 1., 2.3)
    x3 = gaussian (1.2, 1., 2.3)

    assert rv.pdf(1.2) == x2
    assert abs(x2- x3) < 0.00000001
    print "all tests passed"

