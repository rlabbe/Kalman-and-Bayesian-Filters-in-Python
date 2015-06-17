# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:09:23 2014

@author: rlabbe
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
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
    #m = np.mean(ys)
    std = np.std(ys)

    in_lims = [x0-in_std*3, x0+in_std*3]
    out_lims = [y-std*3, y+std*3]



    #plot output
    h = np.histogram(ys, num_bins, density=False)
    plt.subplot(2,2,4)
    plt.plot(h[0], h[1][1:], lw=4, alpha=0.5)
    print(max(h[0]))
    plt.ylim(out_lims[1], out_lims[0])
    plt.gca().xaxis.set_ticklabels([])
    plt.title('output')

    plt.axhline(np.mean(ys), ls='--', lw=2)
    plt.axhline(f(x0), lw=1)


    norm = scipy.stats.norm(y, in_std)

    min_x = norm.ppf(0.001)
    max_x = norm.ppf(0.999)
    xs = np.arange(min_x, max_x, (max_x - min_x) / 1000)
    pdf = norm.pdf(xs)
    plt.plot(pdf * max(h[0])/max(pdf), xs, lw=1, color='k')
    print(max(norm.pdf(xs)))
    return


    # plot transfer function
    plt.subplot(2,2,3)
    x = np.arange(in_lims[0], in_lims[1], 0.1)
    y = f(x)
    plt.plot (x,y, 'k')
    isct = f(x0)
    plt.plot([x0, x0, in_lims[1]], [out_lims[1], isct, isct], color='r', lw=1)
    plt.xlim(in_lims)
    plt.ylim(out_lims)
    #plt.axis('equal')
    plt.title('function')

    # plot input
    h = np.histogram(data, num_bins, density=True)

    plt.subplot(2,2,1)
    plt.plot(h[1][1:], h[0], lw=4)
    plt.xlim(in_lims)
    plt.gca().yaxis.set_ticklabels([])
    plt.title('input')

    plt.show()
    print("fuck")



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


test_plot()
1/0

if __name__ == "__main__":
    from numpy.random import normal
    import numpy as np


    x0 = (1, 1)
    data = normal(loc=x0[0], scale=x0[1], size=500000)

    def g(x):
        return x*x
        return (np.cos(3*(x/2+0.7)))*np.sin(0.7*x)-1.6*x
        return -2*x


    #plot_transfer_func (data, g, lims=(-3,3), num_bins=100)
    plot_nonlinear_func (data, g, gaussian=x0,
                        num_bins=100)
