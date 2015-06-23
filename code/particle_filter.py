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


class ParticleFilter(object):

    def __init__(self, N, x_range, y_range):
        self.particles = np.zeros((N, 4))
        self.N = N
        self.x_range = x_range
        self.y_range = y_range

        # assign
        self.weights = np.array([1./N] * N)
        self.particles[:, 0] = uniform(0, x_range, size=N)
        self.particles[:, 1] = uniform(0, y_range, size=N)
        self.particles[:, 3] = uniform(0, 2*np.pi, size=N)



    def create_particles(self, mean, variance):
        """ create particles with the specied mean and variance"""
        self.particles[:, 0] = mean[0] + randn(self.N) * np.sqrt(variance)
        self.particles[:, 1] = mean[1] + randn(self.N) * np.sqrt(variance)

    def create_particle(self):
        """ create particles uniformly distributed over entire space"""
        return [uniform(0, self.x_range), uniform(0, self.y_range), 0, 0]


    def assign_speed_by_gaussian(self, speed, var):
        """ move every particle by the specified speed (assuming time=1.)
        with the specified variance, assuming Gaussian distribution. """

        self.particles[:, 2] = np.random.normal(speed, var, self.N)

    def control(self, dx):
        self.particles[:, 0] += dx[0]
        self.particles[:, 1] += dx[1]


    def move(self, hdg, vel, t=1.):
        """ move the particles according to their speed and direction for the
        specified time duration t"""
        h = math.atan2(hdg[1], hdg[0])
        h = randn(self.N) * .4 + h
        vs = vel + randn(self.N) * 0.1
        vx = vel * np.cos(h)
        vy = vel * np.sin(h)

        self.particles[:, 0] = (self.particles[:, 0] + vx*t)
        self.particles[:, 1] = (self.particles[:, 1] + vy*t)


    def move2(self, u):
        """ move according to control input u"""

        dx = u[0] + randn(self.N) * 1.9
        dy = u[1] + randn(self.N) * 1.9
        self.particles[:, 0] = (self.particles[:, 0] + dx)
        self.particles[:, 1] = (self.particles[:, 1] + dy)


    def weight(self, z, var):
        dist = np.sqrt((self.particles[:, 0] - z[0])**2 +
                       (self.particles[:, 1] - z[1])**2)

        # simplification assumes variance is invariant to world projection
        n = scipy.stats.norm(0, np.sqrt(var))
        prob = n.pdf(dist)

        # particles far from a measurement will give us 0.0 for a probability
        # due to floating point limits. Once we hit zero we can never recover,
        # so add some small nonzero value to all points.
        prob += 1.e-12
        self.weights *= prob
        self.weights /= sum(self.weights) # normalize


    def neff(self):
        return 1. / np.sum(np.square(self.weights))


    def resample(self):
        p = np.zeros((self.N, 4))
        w = np.zeros(self.N)

        cumsum = np.cumsum(self.weights)
        for i in range(self.N):
            index = np.searchsorted(cumsum, random.random())
            p[i] = self.particles[index]
            w[i] = self.weights[index]

        self.particles = p
        self.weights = w / np.sum(w)


    def estimate(self):
        """ returns mean and variance """
        pos = self.particles[:, 0:2]
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var


def plot(pf, xlim=100, ylim=100, weights=True):

    if weights:
        a = plt.subplot(221)
        a.cla()
        plt.xlim(0, ylim)
        plt.ylim(0, 1)
        plt.scatter(pf.particles[:, 0], pf.weights, marker='.', s=1)
        a = plt.subplot(224)
        a.cla()
        plt.scatter(pf.weights, pf.particles[:, 1], marker='.', s=1)
        plt.ylim(0, xlim)
        plt.xlim(0, 1)

        a = plt.subplot(223)
        a.cla()
    else:
        plt.cla()
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], marker='.', s=1)
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)




if __name__ == '__main__':
    pf = ParticleFilter(5000, 100, 100)
    pf.particles[:,3] = np.random.randn(pf.N)*np.radians(10) + np.radians(45)

    z = np.array([20, 20])
    pf.create_particles(mean=z, variance=40)

    mu0 = np.array([0., 0.])
    plot(pf, weights=False)

    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    for x in range(50):

        z[0] += 1.0 + randn()*0.3
        z[1] += 1.0 + randn()*0.3


        pf.move2((1,1))
        pf.weight(z, 5.2)
#        pf.weight((z[0] + randn()*0.2, z[1] + randn()*0.2), 5.2)
        pf.resample()
        mu, var = pf.estimate()
        if x == 0:
            mu0 = mu
        print(mu - z)
        print('neff', pf.neff())
        #print(var)

        plot(pf, weights=False)
        plt.plot(z[0], z[1], marker='v', c='r', ms=10)
        plt.scatter(mu[0], mu[1], c='g', s=100)#,
                    #s=min(500, abs((1./np.sum(var)))*20), alpha=0.5)
        plt.tight_layout()
        fig.canvas.draw()

        #pf.assign_speed_by_gaussian(1, 1.5)
        #pf.move(h=[1,1], v=1.4, t=1)
        #pf.control(mu-mu0)
        mu0 = mu





