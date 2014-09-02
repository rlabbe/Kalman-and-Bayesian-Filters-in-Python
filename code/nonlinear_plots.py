# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:09:23 2014

@author: rlabbe
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal


def plot_transfer_func(data, f, lims,num_bins=1000):
    ys = f(data)

    #plot output
    plt.subplot(2,2,1)
    plt.hist(ys, num_bins, orientation='horizontal',histtype='step')
    plt.ylim(lims)
    plt.gca().xaxis.set_ticklabels([])
    plt.title('output')

    # plot transfer function
    plt.subplot(2,2,2)
    x = np.arange(lims[0], lims[1],0.1)
    y = f(x)
    plt.plot (x,y)
    isct = f(0)
    plt.plot([0,0,lims[0]],[lims[0],isct,isct],c='r')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title('transfer function')

    # plot input
    plt.subplot(2,2,4)
    plt.hist(data, num_bins, histtype='step')
    plt.xlim(lims)
    plt.gca().yaxis.set_ticklabels([])
    plt.title('input')

    plt.show()
