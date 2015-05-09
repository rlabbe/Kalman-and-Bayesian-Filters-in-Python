# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:09:23 2014

@author: rlabbe
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def plot_transfer_func(data, f, gaussian, num_bins=300):
    ys = f(data)
    x0 = gaussian[0]
    in_std = np.sqrt(gaussian[1])
    y = f(x0)
    m = np.mean(ys)
    std = np.std(ys)
    
    in_lims = [x0-in_std*3, x0+in_std*3]
    out_lims = [y-std*3, y+std*3]
    

   
    #plot output
    h = np.histogram(ys, num_bins, density=False)
    plt.subplot(2,2,4)
    plt.plot(h[0], h[1][1:], lw=4)
    plt.ylim(out_lims[1], out_lims[0])
    plt.gca().xaxis.set_ticklabels([])
    plt.title('output')

    plt.axhline(np.mean(ys), ls='--', lw=2)
    plt.axhline(f(x0), lw=1)

    # plot transfer function
    plt.subplot(2,2,3)
    x = np.arange(in_lims[0], in_lims[1], 0.1)
    y = f(x)
    plt.plot (x,y)
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
    plot_transfer_func (data, g, gaussian=x0,                        
                        num_bins=100)
    