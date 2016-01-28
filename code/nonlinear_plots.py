# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.


Code supporting the book

Kalman and Bayesian Filters in Python
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


This is licensed under an MIT license. See the LICENSE.txt file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform
from filterpy.stats import multivariate_gaussian
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import normal, multivariate_normal
import scipy.stats

def plot_nonlinear_func(data, f, gaussian, num_bins=300):

    # linearize at mean to simulate EKF
    #x = gaussian[0]

    # equation of linearization
    #m = df(x)
    #b = f(x) - x*m

    # compute new mean and variance based on EKF equations
    ys = f(data)
    x0 = gaussian[0]
    in_std = np.sqrt(gaussian[1])
    y = f(x0)
    std = np.std(ys)

    in_lims = [x0-in_std*3, x0+in_std*3]
    out_lims = [y-std*3, y+std*3]


    #plot output
    h = np.histogram(ys, num_bins, density=False)
    plt.subplot(2,2,4)
    plt.plot(h[0], h[1][1:], lw=4, alpha=0.8)
    plt.ylim(out_lims[1], out_lims[0])
    plt.gca().xaxis.set_ticklabels([])
    plt.title('Output')

    plt.axhline(np.mean(ys), ls='--', lw=2)
    plt.axhline(f(x0), lw=1)


    norm = scipy.stats.norm(y, in_std)

    '''min_x = norm.ppf(0.001)
    max_x = norm.ppf(0.999)
    xs = np.arange(min_x, max_x, (max_x - min_x) / 1000)
    pdf = norm.pdf(xs)
    plt.plot(pdf * max(h[0])/max(pdf), xs, lw=1, color='k')
    print(max(norm.pdf(xs)))'''

    # plot transfer function
    plt.subplot(2, 2, 3)
    x = np.arange(in_lims[0], in_lims[1], 0.1)
    y = f(x)
    plt.plot (x, y, 'k')
    isct = f(x0)
    plt.plot([x0, x0, in_lims[1]], [out_lims[1], isct, isct], color='r', lw=1)
    plt.xlim(in_lims)
    plt.ylim(out_lims)
    #plt.axis('equal')
    plt.title('f(x)')

    # plot input
    h = np.histogram(data, num_bins, density=True)

    plt.subplot(2,2,1)
    plt.plot(h[1][1:], h[0], lw=4)
    plt.xlim(in_lims)
    plt.gca().yaxis.set_ticklabels([])
    plt.title('Input')

    plt.show()


import math
def plot_ekf_vs_mc():

    def fx(x):
        return x**3

    def dfx(x):
        return 3*x**2

    mean = 1
    var = .1
    std = math.sqrt(var)

    data = normal(loc=mean, scale=std, size=50000)
    d_t = fx(data)

    mean_ekf = fx(mean)
    slope = dfx(mean)
    std_ekf = abs(slope*std)

    norm = scipy.stats.norm(mean_ekf, std_ekf)
    xs = np.linspace(-3, 5, 200)
    plt.plot(xs, norm.pdf(xs), lw=2, ls='--', color='b')
    plt.hist(d_t, bins=200, normed=True, histtype='step', lw=2, color='g')

    actual_mean = d_t.mean()
    plt.axvline(actual_mean, lw=2, color='g', label='Monte Carlo')
    plt.axvline(mean_ekf, lw=2, ls='--', color='b', label='EKF')
    plt.legend()
    plt.show()

    print('actual mean={:.2f}, std={:.2f}'.format(d_t.mean(), d_t.std()))
    print('EKF    mean={:.2f}, std={:.2f}'.format(mean_ekf, std_ekf))


def plot_ukf_vs_mc(alpha=0.001, beta=3., kappa=1.):

    def fx(x):
        return x**3

    def dfx(x):
        return 3*x**2

    mean = 1
    var = .1
    std = math.sqrt(var)

    data = normal(loc=mean, scale=std, size=50000)
    d_t = fx(data)


    points = MerweScaledSigmaPoints(1, alpha, beta, kappa)
    Wm, Wc = points.weights()
    sigmas = points.sigma_points(mean, var)

    sigmas_f = np.zeros((3, 1))
    for i in range(3):
        sigmas_f[i] = fx(sigmas[i, 0])

    ### pass through unscented transform
    ukf_mean, ukf_cov = unscented_transform(sigmas_f, Wm, Wc)
    ukf_mean = ukf_mean[0]
    ukf_std = math.sqrt(ukf_cov[0])

    norm = scipy.stats.norm(ukf_mean, ukf_std)
    xs = np.linspace(-3, 5, 200)
    plt.plot(xs, norm.pdf(xs), ls='--', lw=2, color='b')
    plt.hist(d_t, bins=200, normed=True, histtype='step', lw=2, color='g')

    actual_mean = d_t.mean()
    plt.axvline(actual_mean, lw=2, color='g', label='Monte Carlo')
    plt.axvline(ukf_mean, lw=2, ls='--', color='b', label='UKF')
    plt.legend()
    plt.show()

    print('actual mean={:.2f}, std={:.2f}'.format(d_t.mean(), d_t.std()))
    print('UKF    mean={:.2f}, std={:.2f}'.format(ukf_mean, ukf_std))



def test_plot():
    import math
    from numpy.random import normal
    from scipy import stats
    global data

    def f(x):
        return 2*x + 1

    mean = 2
    var = 3
    std = math.sqrt(var)

    data = normal(loc=2, scale=std, size=50000)

    d2 = f(data)
    n = scipy.stats.norm(mean, std)

    kde1 = stats.gaussian_kde(data,  bw_method='silverman')
    kde2 = stats.gaussian_kde(d2,  bw_method='silverman')
    xs = np.linspace(-10, 10, num=200)

    #plt.plot(data)
    plt.plot(xs, kde1(xs))
    plt.plot(xs, kde2(xs))
    plt.plot(xs, n.pdf(xs), color='k')

    num_bins=100
    h = np.histogram(data, num_bins, density=True)
    plt.plot(h[1][1:], h[0], lw=4)

    h = np.histogram(d2, num_bins, density=True)
    plt.plot(h[1][1:], h[0], lw=4)



def plot_bivariate_colormap(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xmin = xs.min()
    xmax = xs.max()
    ymin = ys.min()
    ymax = ys.max()
    values = np.vstack([xs, ys])
    kernel = scipy.stats.gaussian_kde(values)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(kernel.evaluate(positions).T, X.shape)
    plt.gca().imshow(np.rot90(Z), cmap=plt.cm.Greys,
                     extent=[xmin, xmax, ymin, ymax])



def plot_monte_carlo_mean(xs, ys, f, mean_fx, label, plot_colormap=True):
    fxs, fys = f(xs, ys)

    computed_mean_x = np.average(fxs)
    computed_mean_y = np.average(fys)

    plt.subplot(121)
    plt.gca().grid(b=False)

    plot_bivariate_colormap(xs, ys)

    plt.scatter(xs, ys, marker='.', alpha=0.02, color='k')
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

    plt.subplot(122)
    plt.gca().grid(b=False)

    plt.scatter(fxs, fys, marker='.', alpha=0.02, color='k')
    plt.scatter(mean_fx[0], mean_fx[1],
                marker='v', s=300, c='r', label=label)
    plt.scatter(computed_mean_x, computed_mean_y,
                marker='*',s=120, c='b', label='Computed Mean')

    plot_bivariate_colormap(fxs, fys)
    plt.ylim([-10, 200])
    plt.xlim([-100, 100])
    plt.legend(loc='best', scatterpoints=1)
    print ('Difference in mean x={:.3f}, y={:.3f}'.format(
           computed_mean_x-mean_fx[0], computed_mean_y-mean_fx[1]))



def plot_cov_ellipse_colormap(cov=[[1,1],[1,1]]):
    side = np.linspace(-3,3,24)
    X,Y = np.meshgrid(side,side)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X;
    pos[:, :, 1] = Y
    plt.axes(xticks=[], yticks=[], frameon=True)
    rv = scipy.stats.multivariate_normal((0,0), cov)
    plt.gca().grid(b=False)
    plt.gca().imshow(rv.pdf(pos), cmap=plt.cm.Greys, origin='lower')
    plt.show()



def plot_gaussians(xs, ps, x_range, y_range, N):
    """ given a list of 2d states (x,y) and 2x2 covariance matrices, produce
    a surface plot showing all of the gaussians"""

    xs = np.asarray(xs)
    x = np.linspace (x_range[0], x_range[1], N)
    y = np.linspace (y_range[0], y_range[1], N)
    xx, yy = np.meshgrid(x, y)
    zv = np.zeros((N, N))

    for mean, cov in zip(xs, ps):
        zs = np.array([multivariate_gaussian(np.array([i ,j]), mean, cov)
                   for i, j in zip(np.ravel(xx), np.ravel(yy))])
        zv += zs.reshape(xx.shape)

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zv, rstride=1, cstride=1, lw=.5, edgecolors='#191919',
                    antialiased=True, shade=True, cmap=cm.autumn)
    ax.view_init(elev=40., azim=230)


if __name__ == "__main__":
    #plot_cov_ellipse_colormap(cov=[[2, 1.2], [1.2, 2]])
    '''
    from numpy.random import normal
    import numpy as np

    plot_ukf_vs_mc()'''

    '''x0 = (1, 1)
    data = normal(loc=x0[0], scale=x0[1], size=500000)

    def g(x):
        return x*x
        return (np.cos(3*(x/2+0.7)))*np.sin(0.7*x)-1.6*x
        return -2*x


    #plot_transfer_func (data, g, lims=(-3,3), num_bins=100)
    plot_nonlinear_func (data, g, gaussian=x0,
                        num_bins=100)
    '''

    Ps = np.array([[[ 2.85841814,  0.71772898],
        [ 0.71772898,  0.93786824]],

       [[ 3.28939458,  0.52634978],
        [ 0.52634978,  0.13435503]],

       [[ 2.40532661,  0.29692055],
        [ 0.29692055,  0.07671416]],

       [[ 2.23084082,  0.27823192],
        [ 0.27823192,  0.07488681]]])

    Ms = np.array([[  0.68040795,   0.17084572],
       [  8.46201389,   1.15070342],
       [ 13.7992229 ,   0.96022707],
       [ 19.95838208,   0.87524265]])

    plot_multiple_gaussians(Ms, Ps, (-5,25), (-5, 5), 75)



