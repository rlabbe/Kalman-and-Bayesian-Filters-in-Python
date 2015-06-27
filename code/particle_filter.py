# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:46:06 2015

@author: Roger
"""

import math
import numpy as np
from numpy.random import uniform
from numpy.random import randn
import scipy.stats
import matplotlib.pyplot as plt
import random












if __name__ == '__main__':
    N = 2000
    pf = ParticleFilter(N, 100, 100)
    #pf.particles[:,2] = np.random.randn(pf.N)*np.radians(10) + np.radians(45)

    z = np.array([20, 20])
    #pf.create_particles(mean=z, variance=40)

    mu0 = np.array([0., 0.])
    plot(pf, weights=False)


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





