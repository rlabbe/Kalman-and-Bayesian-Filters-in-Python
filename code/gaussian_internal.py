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

import filterpy.stats as stats
import math
import matplotlib.pyplot as plt
import numpy as np

def plot_height_std(x, lw=10):
    m = np.mean(x)
    s = np.std(x)

    for i, height in enumerate(x):
        plt.plot([i+1, i+1], [0, height], color='k', lw=lw)
    plt.xlim(0,len(x)+1)
    plt.axhline(m-s, ls='--')
    plt.axhline(m+s, ls='--')
    plt.fill_between((0, len(x)+1), m-s, m+s,
                     facecolor='yellow', alpha=0.4)
    plt.xlabel('student')
    plt.ylabel('height (m)')


def plot_correlated_data(X, Y, xlabel=None,
                         ylabel=None, equal=True):

    plt.scatter(X, Y)

    if xlabel is not None:
        plt.xlabel('Height (in)');

    if ylabel is not None:
        plt.ylabel('Weight (lbs)')

    # fit line through data
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X, np.asarray(X)*m + b,color='k')
    if equal:
        plt.gca().set_aspect('equal')
    plt.show()

def plot_gaussian (mu, variance,
                   mu_line=False,
                   xlim=None,
                   xlabel=None,
                   ylabel=None):

    xs = np.arange(mu-variance*2,mu+variance*2,0.1)
    ys = [stats.gaussian (x, mu, variance)*100 for x in xs]
    plt.plot (xs, ys)
    if mu_line:
        plt.axvline(mu)
    if xlim:
        plt.xlim(xlim)
    if xlabel:
       plt.xlabel(xlabel)
    if ylabel:
       plt.ylabel(ylabel)
    plt.show()

def display_stddev_plot():
    xs = np.arange(10,30,0.1)
    var = 8;
    stddev = math.sqrt(var)
    p2, = plt.plot (xs,[stats.gaussian(x, 20, var) for x in xs])
    x = 20+stddev
    # 1std vertical lines
    y = stats.gaussian(x, 20, var)
    plt.plot ([x,x], [0,y],'g')
    plt.plot ([20-stddev, 20-stddev], [0,y], 'g')

    #2std vertical lines
    x = 20+2*stddev
    y = stats.gaussian(x, 20, var)
    plt.plot ([x,x], [0,y],'g')
    plt.plot ([20-2*stddev, 20-2*stddev], [0,y], 'g')

    y = stats.gaussian(20,20,var)
    plt.plot ([20,20],[0,y],'b')

    x = 20+stddev
    ax = plt.axes()
    ax.annotate('68%', xy=(20.3, 0.045))
    ax.annotate('', xy=(20-stddev,0.04), xytext=(x,0.04),
                arrowprops=dict(arrowstyle="<->",
                                ec="r",
                                shrinkA=2, shrinkB=2))
    ax.annotate('95%', xy=(20.3, 0.02))
    ax.annotate('', xy=(20-2*stddev,0.015), xytext=(20+2*stddev,0.015),
                arrowprops=dict(arrowstyle="<->",
                                ec="r",
                                shrinkA=2, shrinkB=2))


    ax.xaxis.set_ticks ([20-2*stddev, 20-stddev, 20, 20+stddev, 20+2*stddev])
    ax.xaxis.set_ticklabels(['$-2\sigma$', '$-1\sigma$','$\mu$','$1\sigma$', '$2\sigma$'])
    ax.yaxis.set_ticks([])
    ax.grid(None, 'both', lw=0)

if __name__ == '__main__':
    display_stddev_plot()