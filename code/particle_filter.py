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


import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from numpy.random import randn
import scipy.stats
import random



if __name__ == '__main__':
    N = 2000
    pf = ParticleFilter(N, 100, 100)
    #pf.particles[:,2] = np.random.randn(pf.N)*np.radians(10) + np.radians(45)

    z = np.array([20, 20])
    #pf.create_particles(mean=z, variance=40)

    mu0 = np.array([0., 0.])
    plt.plot(pf, weights=False)


    fig = plt.gcf()
    #fig.show()
    #fig.canvas.draw()
    #plt.ioff()

    for x in range(10):

        z[0] = x+1 + randn()*0.3
        z[1] = x+1 + randn()*0.3


        pf.predict((1,1), (0.2, 0.2))
        pf.weight(z=z, var=.8)
        neff = pf.neff()

        print('neff', neff)
        if neff < N/2 or N <= 2000:
            pf.resample()
        mu, var = pf.estimate()
        if x == 0:
            mu0 = mu
        #print(mu - z)
        #print(var)

        plot(pf, weights=True)
        #plt.plot(z[0], z[1], marker='v', c='r', ms=10)
        plt.plot(x+1, x+1, marker='*', c='r', ms=10)
        plt.scatter(mu[0], mu[1], c='g', s=100)#,
                    #s=min(500, abs((1./np.sum(var)))*20), alpha=0.5)
        plt.plot([0,100], [0,100])
        plt.tight_layout()
        plt.pause(.002)

        #fig.canvas.draw()

        #pf.assign_speed_by_gaussian(1, 1.5)
        #pf.move(h=[1,1], v=1.4, t=1)
        #pf.control(mu-mu0)
        mu0 = mu

    plt.ion()





