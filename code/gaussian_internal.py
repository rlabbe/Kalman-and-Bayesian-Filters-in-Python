# -*- coding: utf-8 -*-
"""
Created on Thu May  8 23:16:31 2014

@author: rlabbe
"""

import math
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import filterpy.stats as stats

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
    var = 8; stddev = math.sqrt(var)
    p2, = plt.plot (xs,[stats.gaussian(x, 20, var) for x in xs])
    x = 20+stddev
    y = stats.gaussian(x, 20, var)
    plt.plot ([x,x], [0,y],'g')
    plt.plot ([20-stddev, 20-stddev], [0,y], 'g')
    y = stats.gaussian(20,20,var)
    plt.plot ([20,20],[0,y],'b')
    ax = plt.axes()
    ax.annotate('68%', xy=(20.3, 0.045))
    ax.annotate('', xy=(20-stddev,0.04), xytext=(x,0.04),
                arrowprops=dict(arrowstyle="<->",
                                ec="r",
                                shrinkA=2, shrinkB=2))
    ax.xaxis.set_ticks ([20-stddev, 20, 20+stddev])
    ax.xaxis.set_ticklabels(['$-\sigma$','$\mu$','$\sigma$'])
    ax.yaxis.set_ticks([])
    plt.show()

if __name__ == '__main__':
    display_stddev_plot()