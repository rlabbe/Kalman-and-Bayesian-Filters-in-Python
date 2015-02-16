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



    # plot transfer function
    plt.subplot(2,2,2)
    x = np.arange(lims[0], lims[1],0.1)
    y = f(x)
    plt.plot (x,y)
    isct = f(0)
    plt.plot([0,0,lims[0]],[lims[0],isct,isct],c='r')
    plt.xlim(lims)


    # plot input
    plt.subplot(2,2,4)
    plt.hist(data, num_bins, histtype='step')
    plt.xlim(lims)
    plt.gca().yaxis.set_ticklabels([])


    plt.show()


normals = normal(loc=0.0, scale=1, size=5000000)

#rint h(normals).sort()


def f(x):
    return 2*x + 1

def g(x):
    return (cos(4*(x/2+0.7)))*sin(0.3*x)-0.9*x
    return (cos(4*(x/3+0.7)))*sin(0.3*x)-0.9*x
    #return -x+1.2*np.sin(0.7*x)+3
    return sin(5-.2*x)

def h(x): return cos(.4*x)*x

plot_transfer_func (normals, g, lims=(-4,4),num_bins=500)
del(normals)

#plt.plot(g(np.arange(-10,10,0.1)))

'''


ys = f(normals)


r = np.linspace (min(normals), max(normals), num_bins)

h= np.histogram(ys, num_bins,density=True)
print h
print len(h[0]), len(h[1][0:-1])

#plot output
plt.subplot(2,2,1)
h = np.histogram(ys, num_bins,normed=True)

p, = plt.plot(h[0],h[1][1:])
plt.ylim((-10,10))
plt.xlim((max(h[0]),0))


# plot transfer function
plt.subplot(2,2,2)
x = np.arange(-10,10)
y = 1.2*x + 1
plt.plot (x,y)
plt.plot([0,0],[-10,f(0)],c='r')
plt.ylim((-10,10))

# plot input
plt.subplot(2,2,4)
h = np.histogram(normals, num_bins,density=True)
plt.plot(h[1][1:],h[0])
plt.xlim((-10,10))


plt.show()
'''